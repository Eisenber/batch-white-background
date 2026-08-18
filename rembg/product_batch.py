"""Sequential batch orchestration and downloadable archive creation."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from PIL.Image import Image as PILImage

from .product_image import (
    BatchOptions,
    ImageInput,
    ProcessingResult,
    decode_product_image,
    process_product_image,
    compose_white_canvas,
)
from .sessions.base import BaseSession


ProgressCallback = Callable[[int, int, str], None]
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EDIT_DIRECTORY_NAME = ".edit"
MANIFEST_NAME = "manifest.json"


@dataclass
class BatchItem:
    source_name: str
    output_name: str | None
    result: ProcessingResult


@dataclass
class BatchResult:
    task_directory: Path
    zip_path: Path
    items: list[BatchItem] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        counts = {"processed": 0, "review": 0, "failed": 0}
        for item in self.items:
            counts[item.result.status] += 1
        return counts


def _source_name(value: ImageInput, index: int) -> str:
    if isinstance(value, (str, Path)):
        return Path(value).name
    filename = getattr(value, "name", None)
    if filename:
        return Path(str(filename)).name
    return f"image_{index:03d}.jpg"


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem.strip() or "image"
    return "".join("_" if char in '\\/:*?"<>|' else char for char in stem)


def _unique_output_name(source_name: str, suffix: str, used_names: set[str]) -> str:
    stem = _safe_stem(source_name)
    candidate = f"{stem}{suffix}.jpg"
    counter = 2
    while candidate.casefold() in used_names:
        candidate = f"{stem}{suffix}_{counter}.jpg"
        counter += 1
    used_names.add(candidate.casefold())
    return candidate


def _write_archive(task_directory: Path) -> Path:
    """Build the download archive, excluding private correction assets."""

    zip_path = task_directory / "safe_white_images.zip"
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for directory_name in ("processed", "review"):
            directory = task_directory / directory_name
            files = sorted(directory.iterdir())
            if not files:
                archive.writestr(f"{directory.name}/", "")
            for file_path in files:
                archive.write(file_path, f"{directory.name}/{file_path.name}")
        archive.write(task_directory / "report.csv", "report.csv")
    return zip_path


def _append_report_step(task_directory: Path, source_name: str, step: str) -> None:
    report_path = task_directory / "report.csv"
    with report_path.open("r", newline="", encoding="utf-8-sig") as report_file:
        rows = list(csv.reader(report_file))
    for row in rows[1:]:
        if row and row[0] == source_name:
            existing = row[5] if len(row) > 5 else ""
            if step not in existing:
                row[5] = "；".join(value for value in (existing, step) if value)
            break
    with report_path.open("w", newline="", encoding="utf-8-sig") as report_file:
        csv.writer(report_file).writerows(rows)


def load_correction_assets(
    task_directory: str | Path, index: int
) -> tuple[PILImage, PILImage, dict]:
    """Load an aligned source and alpha mask for one local batch item."""

    task = Path(task_directory)
    manifest = json.loads((task / MANIFEST_NAME).read_text(encoding="utf-8"))
    items = manifest["items"]
    if not 0 <= index < len(items):
        raise IndexError("图片序号无效")
    item = items[index]
    if not item.get("editable"):
        raise ValueError("这张图片处理失败，没有可修正的蒙版")
    edit_directory = task / EDIT_DIRECTORY_NAME
    source = Image.open(edit_directory / f"source_{index:03d}.png").convert("RGB")
    mask = Image.open(edit_directory / f"mask_{index:03d}.png").convert("L")
    return source, mask, manifest


def apply_local_mask_correction(
    task_directory: str | Path,
    index: int,
    painted_mask: np.ndarray,
    mode: str,
) -> tuple[PILImage, Path]:
    """Apply a user-painted delete/restore mask and rebuild the local ZIP."""

    if mode not in {"delete", "restore"}:
        raise ValueError("修正模式必须是 delete 或 restore")
    task = Path(task_directory)
    source_image, mask_image, manifest = load_correction_assets(task, index)
    source = np.asarray(source_image, dtype=np.uint8)
    mask = np.asarray(mask_image, dtype=np.uint8).copy()
    paint = np.asarray(painted_mask)
    if paint.ndim == 3:
        paint = paint.max(axis=2)
    paint = cv2.resize(
        paint.astype(np.uint8),
        (mask.shape[1], mask.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    selected = paint >= 32
    if not selected.any():
        raise ValueError("请先在图片上涂抹需要修正的区域")
    if mode == "delete":
        mask[selected] = 0
        step = "人工删除蒙版区域"
    else:
        mask[selected] = 255
        step = "人工恢复蒙版区域"

    options_data = manifest["options"]
    output, _ = compose_white_canvas(
        source,
        mask,
        int(options_data["output_size"]),
        float(options_data["subject_ratio"]),
    )
    item = manifest["items"][index]
    output_path = task / item["output_path"]
    output_path.write_bytes(
        ProcessingResult(image=output, status="processed").to_jpeg_bytes(
            int(options_data["jpeg_quality"])
        )
    )
    Image.fromarray(mask, mode="L").save(
        task / EDIT_DIRECTORY_NAME / f"mask_{index:03d}.png"
    )
    _append_report_step(task, item["source_name"], step)
    return output, _write_archive(task)


def make_comparison(source: ImageInput, result: ProcessingResult) -> PILImage:
    """Create a compact before/after card without modifying either image."""

    original = decode_product_image(source)
    output = result.image or Image.new("RGB", (1000, 1000), "white")
    card = Image.new("RGB", (1000, 540), (238, 240, 243))
    draw = ImageDraw.Draw(card)
    for column, (image, label) in enumerate(((original, "原图"), (output, "白底结果"))):
        preview = ImageOps.contain(image.convert("RGB"), (470, 450))
        x = column * 500 + (500 - preview.width) // 2
        y = 35 + (450 - preview.height) // 2
        card.paste(preview, (x, y))
        draw.text((column * 500 + 20, 10), label, fill=(32, 38, 48))
    return card


def process_product_batch(
    inputs: Sequence[ImageInput],
    options: BatchOptions,
    session: BaseSession,
    progress: ProgressCallback | None = None,
) -> BatchResult:
    """Process up to fifty images and package outputs plus an audit report."""

    if not inputs:
        raise ValueError("请至少选择一张图片")
    if len(inputs) > 50:
        raise ValueError("每批最多处理 50 张图片")
    unsupported = [
        _source_name(source, index)
        for index, source in enumerate(inputs, start=1)
        if Path(_source_name(source, index)).suffix.lower()
        not in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if unsupported:
        raise ValueError("仅支持 PNG、JPEG/JPG 图片：" + "、".join(unsupported[:3]))
    options.validate()

    task_directory = Path(tempfile.mkdtemp(prefix="rembg-safe-batch-"))
    processed_directory = task_directory / "processed"
    review_directory = task_directory / "review"
    edit_directory = task_directory / EDIT_DIRECTORY_NAME
    processed_directory.mkdir()
    review_directory.mkdir()
    edit_directory.mkdir()
    used_names: set[str] = set()
    items: list[BatchItem] = []

    try:
        total = len(inputs)
        for index, source in enumerate(inputs, start=1):
            source_name = _source_name(source, index)
            if progress:
                progress(index - 1, total, source_name)
            result = process_product_image(source, options, session)
            output_name: str | None = None
            if result.image is not None:
                suffix = "_white" if result.status == "processed" else "_review"
                output_name = _unique_output_name(source_name, suffix, used_names)
                destination_directory = (
                    processed_directory
                    if result.status == "processed"
                    else review_directory
                )
                (destination_directory / output_name).write_bytes(
                    result.to_jpeg_bytes(options.jpeg_quality)
                )
                if result.working_image is not None and result.working_mask is not None:
                    Image.fromarray(result.working_image, mode="RGB").save(
                        edit_directory / f"source_{index - 1:03d}.png",
                        optimize=True,
                    )
                    Image.fromarray(result.working_mask, mode="L").save(
                        edit_directory / f"mask_{index - 1:03d}.png",
                        optimize=True,
                    )
                    result.working_image = None
                    result.working_mask = None
            items.append(BatchItem(source_name, output_name, result))

        report_path = task_directory / "report.csv"
        with report_path.open("w", newline="", encoding="utf-8-sig") as report_file:
            writer = csv.writer(report_file)
            writer.writerow(
                [
                    "原文件名",
                    "输出文件名",
                    "状态",
                    "质量档",
                    "耗时（秒）",
                    "处理步骤",
                    "警告或错误",
                ]
            )
            for item in items:
                result = item.result
                issue = result.error or "；".join(result.warnings)
                writer.writerow(
                    [
                        item.source_name,
                        item.output_name or "",
                        result.status,
                        options.quality,
                        f"{result.elapsed_seconds:.2f}",
                        "；".join(result.applied_steps),
                        issue,
                    ]
                )

        manifest = {
            "options": {
                "output_size": options.output_size,
                "subject_ratio": options.subject_ratio,
                "jpeg_quality": options.jpeg_quality,
            },
            "items": [
                {
                    "source_name": item.source_name,
                    "output_name": item.output_name,
                    "output_path": (
                        f"{item.result.status if item.result.status == 'review' else 'processed'}/"
                        f"{item.output_name}"
                        if item.output_name
                        else None
                    ),
                    "status": item.result.status,
                    "editable": bool(
                        item.output_name
                        and (edit_directory / f"source_{index:03d}.png").exists()
                    ),
                }
                for index, item in enumerate(items)
            ],
        }
        (task_directory / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        zip_path = _write_archive(task_directory)

        if progress:
            progress(total, total, "处理完成")
        return BatchResult(task_directory, zip_path, items)
    except Exception:
        cleanup_batch_directory(task_directory)
        raise


def cleanup_batch_directory(path: str | Path | None) -> None:
    """Delete only task directories created by this module."""

    if not path:
        return
    candidate = Path(path).resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    if (
        candidate.parent == temp_root
        and candidate.name.startswith("rembg-safe-batch-")
        and candidate.is_dir()
    ):
        shutil.rmtree(candidate)


def cleanup_stale_batches() -> None:
    temp_root = Path(tempfile.gettempdir())
    for candidate in temp_root.glob("rembg-safe-batch-*"):
        cleanup_batch_directory(candidate)
