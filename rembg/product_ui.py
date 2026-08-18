"""Gradio workbench for batch safe-product photography."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import gradio as gr
import numpy as np
from PIL import Image

from .product_batch import (
    cleanup_batch_directory,
    cleanup_stale_batches,
    apply_local_mask_correction,
    load_correction_assets,
    process_product_batch,
)
from .product_image import BatchOptions, decode_product_image
from .sessions.base import BaseSession


SessionProvider = Callable[[str], BaseSession]


WORKBENCH_CSS = """
:root {
  --safe-ink: #171b1f;
  --safe-steel: #59636d;
  --safe-bench: #eef1f2;
  --safe-orange: #ed6a20;
  --safe-white: #ffffff;
}
.gradio-container {
  font-family: "PingFang SC", "Microsoft YaHei", system-ui, sans-serif !important;
  background: var(--safe-bench) !important;
  color: var(--safe-ink) !important;
}
.safe-shell { max-width: 1180px; margin: 0 auto; }
.safe-hero {
  background: var(--safe-ink);
  color: var(--safe-white);
  border-radius: 14px;
  padding: 28px 30px 24px;
  box-shadow: 0 12px 30px rgba(23, 27, 31, .13);
}
.safe-kicker {
  color: #ff9a62;
  font: 700 12px/1 "DIN Alternate", "SFMono-Regular", monospace;
  letter-spacing: .18em;
  text-transform: uppercase;
}
.safe-hero h1 { margin: 10px 0 8px; font-size: clamp(26px, 4vw, 42px); letter-spacing: -.035em; }
.safe-hero p { margin: 0; max-width: 720px; color: #c9ced3; line-height: 1.7; }
.safe-plate {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1px;
  margin-top: 22px;
  background: #3a4249;
  border: 1px solid #3a4249;
  border-radius: 8px;
  overflow: hidden;
}
.safe-plate div { background: #23292e; padding: 12px 14px; }
.safe-plate b { display: block; font: 700 17px/1.2 "DIN Alternate", "SFMono-Regular", monospace; }
.safe-plate span { display: block; margin-top: 4px; color: #919aa2; font-size: 12px; }
.safe-panel { background: white; border: 1px solid #d9dee1; border-radius: 12px; padding: 8px; }
.safe-section h3 {
  color: var(--safe-ink) !important;
  font-size: 18px !important;
  letter-spacing: -.01em;
  margin-top: 12px !important;
}
.safe-controls {
  background: var(--safe-white) !important;
  border: 1px solid #d9dee1 !important;
  border-radius: 12px !important;
  padding: 14px !important;
}
.safe-controls, .safe-controls * { color: var(--safe-ink) !important; }
.safe-controls fieldset { background: var(--safe-white) !important; border-color: #d9dee1 !important; }
.safe-controls fieldset label { background: #f3f5f5 !important; border-color: #d9dee1 !important; }
.safe-controls fieldset label.selected { background: #fff0e7 !important; border-color: var(--safe-orange) !important; }
.safe-run button { background: var(--safe-orange) !important; border-color: var(--safe-orange) !important; }
.safe-status {
  background: var(--safe-white);
  border-left: 4px solid var(--safe-orange);
  border-radius: 0 8px 8px 0;
  padding: 10px 14px;
}
.safe-status, .safe-status * { color: var(--safe-ink) !important; }
.safe-results > label, .safe-data > label { color: var(--safe-ink) !important; }
.safe-preview-title {
  background: var(--safe-ink);
  border-left: 5px solid var(--safe-orange);
  border-radius: 7px 7px 0 0;
  padding: 8px 12px 6px;
}
.safe-preview-title h4 { color: var(--safe-white) !important; margin: 0 !important; }
.safe-data { font-family: "DIN Alternate", "SFMono-Regular", monospace !important; }
@media (max-width: 640px) {
  .safe-hero { padding: 22px 20px; border-radius: 10px; }
  .safe-plate { grid-template-columns: 1fr; }
}
"""


def _status_markdown(counts: dict[str, int], total: int) -> str:
    return (
        "### 本批处理完成\n"
        f"共 **{total}** 张 · 成品 **{counts['processed']}** · "
        f"需复核 **{counts['review']}** · 失败 **{counts['failed']}**"
    )


def _painted_pixels(editor_value) -> np.ndarray:
    if not editor_value or not editor_value.get("layers"):
        raise gr.Error("请先用红色画笔涂抹需要修正的区域。")
    painted: np.ndarray | None = None
    for layer in editor_value["layers"]:
        array = np.asarray(layer)
        if array.ndim == 3 and array.shape[2] == 4:
            alpha = array[:, :, 3]
        elif array.ndim == 3:
            alpha = np.any(array[:, :, :3] != 0, axis=2).astype(np.uint8) * 255
        else:
            alpha = array.astype(np.uint8)
        painted = alpha if painted is None else np.maximum(painted, alpha)
    if painted is None or not (painted >= 32).any():
        raise gr.Error("没有检测到画笔区域，请在原图上涂抹后再应用。")
    return painted


def _generated_gallery_from_task(task_directory: str | Path):
    task = Path(task_directory)
    manifest = json.loads((task / "manifest.json").read_text(encoding="utf-8"))
    labels = {"processed": "成品", "review": "需复核", "failed": "失败"}
    gallery = []
    for item in manifest["items"]:
        if item.get("output_path"):
            with Image.open(task / item["output_path"]) as image:
                gallery.append(
                    (
                        image.convert("RGB").copy(),
                        f"{labels[item['status']]} · {item['source_name']}",
                    )
                )
    return gallery


def create_product_ui(session_provider: SessionProvider) -> gr.Blocks:
    """Build the product workbench without changing the existing HTTP API."""

    cleanup_stale_batches()

    def run_batch(files, quality, previous_task, progress=gr.Progress()):
        if not files:
            raise gr.Error("请先拖入 1–50 张保险柜图片。")
        if len(files) > 50:
            raise gr.Error("每批最多处理 50 张图片，请拆分后重试。")
        cleanup_batch_directory(previous_task)
        options = BatchOptions(quality=quality)
        try:
            session = session_provider(options.model_name)

            def update(done: int, total: int, filename: str) -> None:
                progress(
                    (done, total),
                    desc=f"正在处理：{filename}" if done < total else filename,
                )

            batch = process_product_batch(files, options, session, update)
        except Exception as exc:
            raise gr.Error(f"批量处理未能启动：{exc}") from exc

        original_gallery = []
        generated_gallery = []
        table = []
        status_labels = {
            "processed": "成品",
            "review": "需复核",
            "failed": "失败",
        }
        for source, item in zip(files, batch.items):
            result = item.result
            issue = result.error or "；".join(result.warnings) or "—"
            original_gallery.append(
                (decode_product_image(source), f"原图 · {item.source_name}")
            )
            if result.image is not None:
                generated_gallery.append(
                    (
                        result.image,
                        f"{status_labels[result.status]} · {item.source_name}",
                    )
                )
            table.append(
                [
                    item.source_name,
                    status_labels[result.status],
                    f"{result.elapsed_seconds:.1f}",
                    "；".join(result.applied_steps),
                    issue,
                ]
            )
        return (
            original_gallery,
            generated_gallery,
            _status_markdown(batch.counts, len(batch.items)),
            table,
            str(batch.zip_path),
            str(batch.task_directory),
        )

    def select_for_correction(task_directory, event: gr.SelectData):
        if not task_directory:
            raise gr.Error("请先完成一批图片处理。")
        raw_index = event.index
        index = int(raw_index[0] if isinstance(raw_index, tuple) else raw_index)
        try:
            source, _, _ = load_correction_assets(task_directory, index)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        editor_value = {
            "background": source,
            "layers": [],
            "composite": source,
        }
        return editor_value, index, f"已选择第 {index + 1} 张，请涂抹需要修正的局部。"

    def apply_correction(task_directory, selected_index, editor_value, mode):
        if not task_directory or selected_index is None:
            raise gr.Error("请先点击左侧“原图”中的一张图片。")
        painted = _painted_pixels(editor_value)
        try:
            output, zip_path = apply_local_mask_correction(
                task_directory, int(selected_index), painted, mode
            )
            source, _, _ = load_correction_assets(task_directory, int(selected_index))
        except Exception as exc:
            raise gr.Error(f"局部修正失败：{exc}") from exc
        action = "删除" if mode == "delete" else "恢复"
        return (
            _generated_gallery_from_task(task_directory),
            output,
            str(zip_path),
            {"background": source, "layers": [], "composite": source},
            f"已应用人工{action}，白底图和 ZIP 已更新。",
        )

    with gr.Blocks(
        title="保险柜白底主图工作台",
        analytics_enabled=False,
        delete_cache=(86400, 86400),
    ) as workbench:
        task_state = gr.State(
            value=None,
            time_to_live=86400,
            delete_callback=cleanup_batch_directory,
        )
        selected_index = gr.State(value=None)
        gr.HTML(
            """
            <div class="safe-shell safe-hero">
              <div class="safe-kicker">SAFE PRODUCT IMAGING / LOCAL</div>
              <h1>保险柜白底主图工作台</h1>
              <p>批量抠图、保真校正与统一排版。斜侧和开门视角保持真实；不可靠的图片自动进入复核区。</p>
              <div class="safe-plate" aria-label="输出规格">
                <div><b>1000 × 1000</b><span>固定方形画布</span></div>
                <div><b>85%</b><span>主体最长边占比</span></div>
                <div><b>LOCAL ONLY</b><span>图片不离开本机</span></div>
              </div>
            </div>
            """
        )
        gr.Markdown("### 1. 选择图片", elem_classes=["safe-section"])
        with gr.Row(equal_height=True):
            files = gr.File(
                label="拖入 PNG、JPEG/JPG（最多 50 张）",
                file_count="multiple",
                file_types=[".png", ".jpg", ".jpeg"],
                type="filepath",
                height=220,
                elem_classes=["safe-panel"],
            )
            with gr.Column(min_width=280, elem_classes=["safe-controls"]):
                quality = gr.Radio(
                    choices=[
                        ("高质量 · BiRefNet Massive", "high"),
                        ("快速 · U2Net", "fast"),
                    ],
                    value="high",
                    label="处理档位",
                    info="高质量档约 27 秒/张；快速档约 9 秒/张（本机实测）。",
                )
                run = gr.Button(
                    "生成白底主图",
                    variant="primary",
                    size="lg",
                    elem_classes=["safe-run"],
                )
                clear = gr.ClearButton(
                    value="清空本批",
                    components=[files],
                    size="md",
                )

        gr.Markdown("### 2. 检查结果", elem_classes=["safe-section"])
        status = gr.Markdown(
            "上传图片后开始处理；单张失败不会中断整批。",
            elem_classes=["safe-status"],
        )
        with gr.Row(equal_height=True):
            with gr.Column():
                gr.Markdown("#### 原图", elem_classes=["safe-preview-title"])
                original_gallery = gr.Gallery(
                    show_label=False,
                    columns=1,
                    object_fit="contain",
                    height="auto",
                    allow_preview=True,
                    buttons=["fullscreen"],
                    elem_classes=["safe-results"],
                )
            with gr.Column():
                gr.Markdown("#### 生成图", elem_classes=["safe-preview-title"])
                generated_gallery = gr.Gallery(
                    show_label=False,
                    columns=1,
                    object_fit="contain",
                    height="auto",
                    allow_preview=True,
                    buttons=["fullscreen"],
                    elem_classes=["safe-results"],
                )
        details = gr.Dataframe(
            headers=["原文件名", "状态", "秒", "处理步骤", "警告或错误"],
            datatype=["str", "str", "number", "str", "str"],
            interactive=False,
            wrap=True,
            label="处理明细",
            elem_classes=["safe-data"],
        )
        with gr.Accordion("局部细节修正（可选）", open=False):
            gr.Markdown(
                "点击上方左侧的一张 **原图**，然后在下方涂抹误保留或误删除的区域。"
                "应用修正只会更新蒙版，不会重新运行抠图模型。"
            )
            correction_status = gr.Markdown("尚未选择图片。")
            with gr.Row():
                editor = gr.ImageEditor(
                    label="在对齐原图上涂抹",
                    type="pil",
                    image_mode="RGBA",
                    sources=(),
                    transforms=(),
                    brush=gr.Brush(
                        default_size=24,
                        colors=["#ff2d20"],
                        default_color="#ff2d20",
                        color_mode="fixed",
                    ),
                    eraser=gr.Eraser(default_size=30),
                    layers=True,
                    format="png",
                    height=520,
                )
                correction_preview = gr.Image(
                    label="修正后的白底图",
                    type="pil",
                    image_mode="RGB",
                    height=520,
                )
            with gr.Row():
                correction_mode = gr.Radio(
                    choices=[("删除多余背景", "delete"), ("恢复主体细节", "restore")],
                    value="delete",
                    label="画笔作用",
                )
                apply_correction_button = gr.Button(
                    "应用局部修正", variant="primary"
                )
        gr.Markdown("### 3. 下载", elem_classes=["safe-section"])
        download = gr.DownloadButton(
            "下载 ZIP（成品、复核图和报告）",
            variant="primary",
        )

        run.click(
            fn=run_batch,
            inputs=[files, quality, task_state],
            outputs=[
                original_gallery,
                generated_gallery,
                status,
                details,
                download,
                task_state,
            ],
            concurrency_limit=1,
            concurrency_id="safe-product-processing",
            api_visibility="private",
        )
        original_gallery.select(
            fn=select_for_correction,
            inputs=[task_state],
            outputs=[editor, selected_index, correction_status],
            api_visibility="private",
        )
        apply_correction_button.click(
            fn=apply_correction,
            inputs=[task_state, selected_index, editor, correction_mode],
            outputs=[
                generated_gallery,
                correction_preview,
                download,
                editor,
                correction_status,
            ],
            concurrency_limit=1,
            concurrency_id="safe-product-processing",
            api_visibility="private",
        )
        clear.add(
            [
                original_gallery,
                generated_gallery,
                status,
                details,
                download,
                task_state,
                selected_index,
                editor,
                correction_preview,
                correction_status,
            ]
        )

    return workbench
