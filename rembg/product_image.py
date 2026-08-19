"""Product-photo processing for safe and lockbox catalog images.

The local path deliberately avoids generative reconstruction. The optional
OpenAI path is isolated behind a complete-image generator and always returns a
review warning because generated pixels can alter product details.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, Union, cast

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps
from PIL.Image import Image as PILImage

from .sessions.base import BaseSession

QualityPreset = Literal["high", "fast"]
OutputFormat = Literal["jpg", "jpeg", "png"]
ProcessingEngine = Literal["local", "openai"]
ResultStatus = Literal["processed", "review", "failed"]
ImageInput = Union[bytes, bytearray, str, Path, PILImage]


@dataclass(frozen=True)
class BatchOptions:
    """Stable options shared by the web UI and programmatic callers."""

    quality: QualityPreset = "high"
    processing_engine: ProcessingEngine = "local"
    jpeg_quality: int = 95
    output_format: OutputFormat = "jpg"
    correct_geometry: bool = True
    correct_glare: bool = True

    @property
    def model_name(self) -> str:
        if self.processing_engine == "openai":
            return "gpt-image-2"
        return "birefnet-massive" if self.quality == "high" else "u2net"

    def validate(self) -> None:
        if self.quality not in ("high", "fast"):
            raise ValueError("quality must be 'high' or 'fast'")
        if self.processing_engine not in ("local", "openai"):
            raise ValueError("processing_engine must be 'local' or 'openai'")
        if not 75 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 75 and 100")
        if self.output_format not in ("jpg", "jpeg", "png"):
            raise ValueError("output_format must be 'jpg', 'jpeg', or 'png'")


@dataclass
class ProcessingResult:
    """Result and audit trail for a single product image."""

    image: PILImage | None
    status: ResultStatus
    applied_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float | str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    error: str | None = None
    # Aligned source pixels and alpha are retained only long enough for the
    # batch layer to persist optional local-mask correction assets.
    working_image: np.ndarray | None = field(default=None, repr=False)
    working_mask: np.ndarray | None = field(default=None, repr=False)

    def to_image_bytes(
        self, output_format: OutputFormat = "jpg", jpeg_quality: int = 95
    ) -> bytes:
        if self.image is None:
            raise ValueError("failed results do not contain an output image")
        output = io.BytesIO()
        image = self.image.convert("RGB")
        if output_format in ("jpg", "jpeg"):
            image.save(
                output,
                format="JPEG",
                quality=jpeg_quality,
                subsampling=0,
                optimize=True,
            )
        elif output_format == "png":
            image.save(output, format="PNG", optimize=True)
        else:
            raise ValueError("output_format must be 'jpg', 'jpeg', or 'png'")
        return output.getvalue()


@dataclass(frozen=True)
class GeneratedProductImage:
    """A complete generated result plus geometry needed for auditing."""

    image: PILImage
    model: str
    requested_width: int
    requested_height: int
    returned_width: int
    returned_height: int


class ProductImageGenerator(Protocol):
    """Generate one complete product image instead of a segmentation mask."""

    def generate(self, image: PILImage) -> GeneratedProductImage: ...


class CloudEnhancer(Protocol):
    """Reserved extension point; version one provides no implementation."""

    def enhance(  # noqa: E704
        self, image: PILImage, result: ProcessingResult
    ) -> ProcessingResult: ...


def decode_product_image(data: ImageInput) -> PILImage:
    """Decode common phone formats and normalize orientation/colour to RGB."""

    if isinstance(data, PILImage):
        image = data.copy()
    elif isinstance(data, (str, Path)):
        image = Image.open(data)
    elif isinstance(data, (bytes, bytearray)):
        image = Image.open(io.BytesIO(bytes(data)))
    else:
        raise TypeError(f"Unsupported image input: {type(data)!r}")

    image.load()
    image = cast(PILImage, ImageOps.exif_transpose(image))

    icc_profile = image.info.get("icc_profile")
    if icc_profile:
        try:
            source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
            target_profile = ImageCms.createProfile("sRGB")
            image = ImageCms.profileToProfile(
                image, source_profile, target_profile, outputMode="RGB"
            )
        except (ImageCms.PyCMSError, OSError, ValueError):
            image = image.convert("RGB")
    else:
        image = image.convert("RGB")

    return cast(PILImage, image)


def _largest_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep probable foreground only when it is anchored to a strong core."""

    probable = (mask >= 32).astype(np.uint8)
    core = (mask >= 160).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(probable, 8)
    if count <= 1:
        return np.zeros_like(mask), 0

    core_labels = set(int(value) for value in np.unique(labels[core > 0]))
    core_labels.discard(0)
    if not core_labels:
        return np.zeros_like(mask), 0

    core_areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in core_labels]
    largest = max(core_areas)
    minimum = max(64, int(largest * 0.002))
    kept = np.zeros_like(probable)
    retained_components = 0
    for index in sorted(core_labels):
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum:
            kept[labels == index] = 1
            retained_components += 1

    soft = mask.copy()
    soft[kept == 0] = 0
    return soft, retained_components


def _top_profile(binary: np.ndarray, left: int, right: int) -> np.ndarray:
    """Return the upper silhouette for each column, interpolating short gaps."""

    height = binary.shape[0]
    profile = np.full(right - left + 1, np.nan, dtype=np.float32)
    for offset, x in enumerate(range(left, right + 1)):
        rows = np.flatnonzero(binary[:, x])
        if rows.size:
            profile[offset] = float(rows[0])
    valid = np.flatnonzero(~np.isnan(profile))
    if valid.size:
        profile = np.interp(np.arange(profile.size), valid, profile[valid]).astype(
            np.float32
        )
    else:
        profile.fill(float(height))
    return profile


def _remove_top_protrusions(
    image: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, int, float, bool]:
    """Remove narrow, colour-inconsistent objects protruding above a safe top.

    This is intentionally conservative.  It operates on upward notches in the
    upper silhouette only when the surrounding silhouette supplies a stable
    top reference and the candidate differs visibly from the cabinet surface.
    """

    binary = (mask >= 128).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return mask, 0, 0.0, False

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    subject = (labels == largest_label).astype(np.uint8)
    x, y, width, height, area = [int(value) for value in stats[largest_label]]
    if width < 160 or height < 120 or area < width * height * 0.18:
        return mask, 0, 0.0, False

    profile = _top_profile(subject, x, x + width - 1)
    # Morphological closing fills narrow upward valleys (objects extending
    # above the cabinet) while retaining broad door/body perspective changes.
    window = max(15, int(round(width * 0.115)))
    if window % 2 == 0:
        window += 1
    reference = cv2.morphologyEx(
        profile.reshape(1, -1),
        cv2.MORPH_CLOSE,
        np.ones((1, window), np.uint8),
    ).reshape(-1)
    reference = cv2.GaussianBlur(reference.reshape(1, -1), (0, 0), 2.0).reshape(-1)

    minimum_depth = max(8.0, height * 0.018)
    raised = (reference - profile) >= minimum_depth
    raised_u8 = raised.astype(np.uint8)
    candidate_count, candidate_labels, candidate_stats, _ = (
        cv2.connectedComponentsWithStats(raised_u8.reshape(1, -1), 8)
    )
    if candidate_count <= 1:
        return mask, 0, 0.0, False

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    refined = mask.copy()
    removed_pixels = 0
    uncertain = False
    maximum_width = max(12, int(round(width * 0.105)))
    maximum_removed = int(round(area * 0.055))

    for candidate_index in range(1, candidate_count):
        start = int(candidate_stats[candidate_index, cv2.CC_STAT_LEFT])
        candidate_width = int(candidate_stats[candidate_index, cv2.CC_STAT_WIDTH])
        if candidate_width < 3 or candidate_width > maximum_width:
            continue
        end = start + candidate_width
        global_left = x + start
        global_right = x + end
        local_reference = reference[start:end]
        local_profile = profile[start:end]
        depth = float(np.max(local_reference - local_profile))
        if depth < minimum_depth:
            continue

        candidate_region = np.zeros_like(subject, dtype=bool)
        comparison_region = np.zeros_like(subject, dtype=bool)
        band_depth = max(5, int(round(height * 0.035)))
        for offset, global_x in enumerate(range(global_left, global_right)):
            boundary = int(round(local_reference[offset]))
            candidate_region[: max(0, boundary - 1), global_x] = True
            comparison_region[
                max(0, boundary + 2) : min(image.shape[0], boundary + band_depth),
                global_x,
            ] = True
        # Clear the complete soft-alpha fringe as well as the opaque core, so
        # removed objects cannot leave a pale outline on the white canvas.
        candidate_region &= mask >= 12
        comparison_region &= subject.astype(bool)
        candidate_area = int(candidate_region.sum())
        if candidate_area < max(20, int(area * 0.00015)):
            continue
        if candidate_area > maximum_removed or comparison_region.sum() < 20:
            uncertain = True
            continue

        candidate_colour = np.median(lab[candidate_region], axis=0)
        surface_colour = np.median(lab[comparison_region], axis=0)
        colour_distance = float(np.linalg.norm(candidate_colour - surface_colour))
        # Matching painted structures are preserved.  Bottles, jars and chair
        # backs in the hard-negative samples are comfortably above this gate.
        if colour_distance < 10.0:
            continue

        removal_region = cv2.dilate(
            candidate_region.astype(np.uint8), np.ones((5, 9), np.uint8)
        ).astype(bool)
        removal_region &= mask > 0
        # Dilation is only for antialiased side fringes; never let it bite into
        # the detected cabinet surface below the reconstructed top line.
        for global_x in range(max(x, global_left - 4), min(x + width, global_right + 4)):
            profile_index = global_x - x
            boundary = int(round(reference[profile_index]))
            removal_region[boundary:, global_x] = False
        refined[removal_region] = 0
        removed_pixels += int(removal_region.sum())

    if removed_pixels > maximum_removed:
        return mask, 0, 0.0, True
    ratio = removed_pixels / max(area, 1)
    return refined, removed_pixels, ratio, uncertain


def _mask_quality(mask: np.ndarray) -> tuple[list[str], dict[str, float]]:
    support = mask >= 24
    height, width = support.shape
    area = int(support.sum())
    total = height * width
    warnings: list[str] = []
    metrics: dict[str, float] = {"mask_area_ratio": area / max(total, 1)}

    if area == 0:
        warnings.append("未检测到完整主体")
        return warnings, metrics

    ys, xs = np.where(support)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    metrics["source_bbox_width"] = float(bbox[2] - bbox[0] + 1)
    metrics["source_bbox_height"] = float(bbox[3] - bbox[1] + 1)

    border_pixels = int(support[0].sum() + support[-1].sum())
    border_pixels += int(support[:, 0].sum() + support[:, -1].sum())
    border_ratio = border_pixels / max(2 * (width + height), 1)
    metrics["border_contact_ratio"] = border_ratio
    if border_ratio > 0.008:
        warnings.append("主体接触原图边缘，可能拍摄不完整")

    area_ratio = area / max(total, 1)
    if area_ratio < 0.025:
        warnings.append("主体面积过小")
    elif area_ratio > 0.92:
        warnings.append("主体或蒙版面积异常")

    return warnings, metrics


def _rotation_angle(
    image: np.ndarray, mask: np.ndarray, minimum_angle: float = 0.6
) -> tuple[float, float]:
    """Return a conservative roll correction and its line-support score."""

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edges[mask < 64] = 0
    minimum = max(40, min(image.shape[:2]) // 10)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        threshold=max(30, minimum // 2),
        minLineLength=minimum,
        maxLineGap=max(8, minimum // 8),
    )
    if lines is None:
        return 0.0, 0.0

    candidates: list[tuple[float, float]] = []
    # OpenCV 5 returns lines as (N, 4); OpenCV 4 wrapped them as (N, 1, 4).
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        angle = float(np.degrees(np.arctan2(dy, dx)))
        normalized = ((angle + 45.0) % 90.0) - 45.0
        if abs(normalized) <= 8.0:
            candidates.append((normalized, length))

    if len(candidates) < 4:
        return 0.0, 0.0

    angles = np.array([item[0] for item in candidates])
    weights = np.array([item[1] for item in candidates])
    order = np.argsort(angles)
    cumulative = np.cumsum(weights[order])
    median_index = order[int(np.searchsorted(cumulative, cumulative[-1] / 2))]
    correction = float(angles[median_index])
    consistency = float(np.average(np.abs(angles - correction) < 1.5, weights=weights))
    if consistency < 0.55 or not minimum_angle <= abs(correction) <= 5.0:
        return 0.0, consistency
    return correction, consistency


def _rotate_pair(
    image: np.ndarray, mask: np.ndarray, angle: float, margin: int = 2
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Rotate around the subject centre on the original canvas when safe."""

    height, width = image.shape[:2]
    support = mask >= 12
    if not support.any():
        return image, mask, False
    ys, xs = np.where(support)
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    corners = np.array(
        [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], dtype=np.float32
    )
    transformed = cv2.transform(corners, matrix)[0]
    if (
        transformed[:, 0].min() < margin
        or transformed[:, 1].min() < margin
        or transformed[:, 0].max() > width - 1 - margin
        or transformed[:, 1].max() > height - 1 - margin
    ):
        return image, mask, False
    rotated_image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    rotated_mask = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated_image, rotated_mask, True


def _safe_perspective_correction(
    image: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rectify only a near-frontal, single-quadrilateral silhouette."""

    binary = (mask >= 128).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, mask, 0.0
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) != 4:
        return image, mask, 0.0

    points = approx[:, 0, :].astype(np.float32)
    center = points.mean(axis=0)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(points.sum(axis=1))]
    ordered[2] = points[np.argmax(points.sum(axis=1))]
    ordered[1] = points[np.argmax(points[:, 0] - points[:, 1])]
    ordered[3] = points[np.argmin(points[:, 0] - points[:, 1])]
    if len({tuple(point) for point in ordered}) != 4:
        return image, mask, 0.0

    top = np.linalg.norm(ordered[1] - ordered[0])
    bottom = np.linalg.norm(ordered[2] - ordered[3])
    left = np.linalg.norm(ordered[3] - ordered[0])
    right = np.linalg.norm(ordered[2] - ordered[1])
    if min(top, bottom, left, right) < 40:
        return image, mask, 0.0

    horizontal_ratio = min(top, bottom) / max(top, bottom)
    vertical_ratio = min(left, right) / max(left, right)
    contour_area = cv2.contourArea(contour)
    quad_area = cv2.contourArea(ordered)
    rectangularity = contour_area / max(quad_area, 1.0)
    if horizontal_ratio < 0.88 or vertical_ratio < 0.88 or rectangularity < 0.88:
        return image, mask, 0.0

    target_width = int(round((top + bottom) / 2.0))
    target_height = int(round((left + right) / 2.0))
    destination = np.array(
        [
            center + (-target_width / 2, -target_height / 2),
            center + (target_width / 2, -target_height / 2),
            center + (target_width / 2, target_height / 2),
            center + (-target_width / 2, target_height / 2),
        ],
        dtype=np.float32,
    )
    displacement = float(
        np.max(np.linalg.norm(destination - ordered, axis=1))
        / max(target_width, target_height)
    )
    if displacement < 0.008 or displacement > 0.06:
        return image, mask, 0.0

    transform = cv2.getPerspectiveTransform(ordered, destination)
    size = (image.shape[1], image.shape[0])
    corrected_image = cv2.warpPerspective(
        image,
        transform,
        size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    corrected_mask = cv2.warpPerspective(
        mask,
        transform,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return corrected_image, corrected_mask, displacement


def _correct_glare(
    image: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Compress broad paint glare while retaining the source high frequencies."""

    support = mask >= 128
    if not support.any():
        return image, 0.0, 0.0

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    luminance = lab[:, :, 0]
    subject_values = luminance[support]
    threshold = max(205.0, float(np.percentile(subject_values, 85)))
    sigma = max(3.0, min(image.shape[:2]) * 0.012)
    base = cv2.GaussianBlur(luminance, (0, 0), sigmaX=sigma, sigmaY=sigma)
    detail = luminance - base

    gradient_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    residual = luminance - cv2.GaussianBlur(luminance, (0, 0), sigmaX=sigma * 2)
    highlight = support & (luminance > threshold)
    highlight &= (residual > 5.0) | (base > threshold + 3.0)
    highlight &= gradient < max(22.0, float(np.percentile(gradient[support], 70)))

    highlight_u8 = highlight.astype(np.uint8) * 255
    radius = max(3, int(round(min(image.shape[:2]) * 0.002)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    highlight_u8 = cv2.morphologyEx(highlight_u8, cv2.MORPH_CLOSE, kernel)
    highlight_u8 = cv2.GaussianBlur(highlight_u8, (0, 0), sigmaX=max(2, radius))
    strength = highlight_u8.astype(np.float32) / 255.0

    compressed_base = np.where(
        base > threshold, threshold + (base - threshold) * 0.55, base
    )
    corrected_luminance = base * (1.0 - strength) + compressed_base * strength + detail
    lab[:, :, 0] = np.clip(corrected_luminance, 0, 255)
    corrected = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    corrected[~support] = image[~support]

    highlight_ratio = float((highlight & support).sum() / max(int(support.sum()), 1))
    clipped = support & (image.max(axis=2) >= 253)
    clipped_ratio = float(clipped.sum() / max(int(support.sum()), 1))
    return corrected, highlight_ratio, clipped_ratio


def _refine_mask_edge(mask: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """Tighten an unusually broad soft fringe without moving its midpoint."""

    binary = (mask >= 128).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    semi_transparent = (mask > 0) & (mask < 255)
    estimated_width = float(semi_transparent.sum() / max(perimeter, 1.0))
    if estimated_width <= 2.0:
        return mask, estimated_width, False

    alpha = mask.astype(np.float32) / 255.0
    normalized = np.clip((alpha - 0.25) / 0.5, 0.0, 1.0)
    refined = normalized * normalized * (3.0 - 2.0 * normalized)
    return np.rint(refined * 255.0).astype(np.uint8), estimated_width, True


def compose_white_canvas(
    image: np.ndarray, mask: np.ndarray
) -> tuple[PILImage, dict[str, float]]:
    """Composite aligned source pixels over white without crop or resize."""

    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask dimensions must match")
    support = mask >= 12
    if not support.any():
        raise ValueError("No foreground subject was detected")

    refined_mask, fringe_width, edge_refined = _refine_mask_edge(mask)
    ys, xs = np.where(support)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    alpha = refined_mask.astype(np.float32)[:, :, None] / 255.0
    canvas = image.astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
    output = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
    return output, {
        "output_width": float(image.shape[1]),
        "output_height": float(image.shape[0]),
        "output_subject_width": float(x1 - x0),
        "output_subject_height": float(y1 - y0),
        "output_subject_center_x": float((x0 + x1 - 1) / 2.0),
        "output_subject_center_y": float((y0 + y1 - 1) / 2.0),
        "output_subject_scale": 1.0,
        "mask_soft_fringe_width": fringe_width,
        "mask_edge_refined": float(edge_refined),
    }


def process_product_image(
    image: ImageInput,
    options: BatchOptions,
    session: BaseSession,
) -> ProcessingResult:
    """Create a source-sized, position-preserving white-background image."""

    started = time.perf_counter()
    options.validate()
    steps = ["方向与颜色标准化"]
    warnings: list[str] = []
    metrics: dict[str, float | str] = {}

    try:
        source = decode_product_image(image)
        rgb = np.asarray(source, dtype=np.uint8)
        masks = session.predict(source)
        if not masks:
            raise ValueError("Background-removal model returned no mask")
        mask = np.asarray(masks[0].convert("L"), dtype=np.uint8)
        metrics["mask_returned_width"] = float(mask.shape[1])
        metrics["mask_returned_height"] = float(mask.shape[0])
        if mask.shape != rgb.shape[:2]:
            mask = cv2.resize(
                mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
            )
            metrics["mask_resized_to_source"] = 1.0
        else:
            metrics["mask_resized_to_source"] = 0.0
        mask, component_count = _largest_mask(mask)
        metrics["mask_component_count"] = float(component_count)
        steps.append(f"主体分割（{options.model_name}）")

        mask, removed_pixels, removed_ratio, cleanup_uncertain = (
            _remove_top_protrusions(rgb, mask)
        )
        metrics["top_cleanup_removed_pixels"] = float(removed_pixels)
        metrics["top_cleanup_removed_ratio"] = removed_ratio
        if removed_pixels:
            steps.append("保险柜顶部杂物清理")
        if cleanup_uncertain:
            warnings.append("检测到疑似顶部杂物但无法安全自动移除")

        quality_warnings, quality_metrics = _mask_quality(mask)
        warnings.extend(quality_warnings)
        metrics.update(quality_metrics)
        if not (mask >= 24).any():
            raise ValueError("未检测到保险柜主体")
        if component_count > 6:
            warnings.append("蒙版包含较多分离区域，请检查边缘")

        if options.correct_geometry:
            angle, support = _rotation_angle(rgb, mask)
            metrics["rotation_line_support"] = support
            metrics["rotation_degrees"] = angle
            metrics["rotation_status"] = "not_needed"
            if angle:
                rgb, mask, rotated = _rotate_pair(rgb, mask, angle)
                if rotated:
                    metrics["rotation_status"] = "applied"
                    steps.append(f"主体原位回正（{angle:+.2f}°）")
                else:
                    metrics["rotation_status"] = "skipped_clipping_risk"
                    metrics["rotation_clipping_risk"] = 1.0
                    warnings.append("主体靠近边缘，为避免裁切已跳过自动回正")
        else:
            metrics["rotation_status"] = "disabled"

        if options.correct_glare:
            rgb, highlight_ratio, clipped_ratio = _correct_glare(rgb, mask)
            metrics["highlight_area_ratio"] = highlight_ratio
            metrics["clipped_highlight_ratio"] = clipped_ratio
            if highlight_ratio > 0.0005:
                steps.append("保纹理高光压制")
            if clipped_ratio > 0.012:
                warnings.append("存在较大过曝区域，缺失纹理无法可靠恢复")

        output, compose_metrics = compose_white_canvas(rgb, mask)
        metrics.update(compose_metrics)
        if compose_metrics["mask_edge_refined"]:
            steps.append("蒙版边缘收紧")
        steps.append("原尺寸原位置白底合成")
        status: ResultStatus = "review" if warnings else "processed"
        return ProcessingResult(
            image=output,
            status=status,
            applied_steps=steps,
            warnings=warnings,
            metrics=metrics,
            elapsed_seconds=time.perf_counter() - started,
            working_image=rgb,
            working_mask=mask,
        )
    except Exception as exc:
        return ProcessingResult(
            image=None,
            status="failed",
            applied_steps=steps,
            warnings=warnings,
            metrics=metrics,
            elapsed_seconds=time.perf_counter() - started,
            error=str(exc),
        )


def process_generated_product_image(
    image: ImageInput,
    options: BatchOptions,
    generator: ProductImageGenerator,
) -> ProcessingResult:
    """Generate a full white-background image and restore source dimensions."""

    started = time.perf_counter()
    options.validate()
    steps = ["方向与颜色标准化"]
    warnings = ["GPT 生成结果需人工复核文字、按键、锁具和表面细节"]
    metrics: dict[str, float | str] = {}

    try:
        if options.processing_engine != "openai":
            raise ValueError("完整图片生成仅支持 openai 处理引擎")
        source = decode_product_image(image)
        metrics["source_width"] = float(source.width)
        metrics["source_height"] = float(source.height)
        generated = generator.generate(source)
        output = generated.image.convert("RGB")
        metrics.update(
            {
                "model": generated.model,
                "requested_width": float(generated.requested_width),
                "requested_height": float(generated.requested_height),
                "returned_width": float(generated.returned_width),
                "returned_height": float(generated.returned_height),
            }
        )
        steps.append("GPT Image 2 高质量白底生成")
        resized = output.size != source.size
        if resized:
            output = output.resize(source.size, Image.Resampling.LANCZOS)
            steps.append("恢复原图尺寸")
        metrics["resized_to_source"] = float(resized)
        metrics["final_width"] = float(output.width)
        metrics["final_height"] = float(output.height)
        return ProcessingResult(
            image=output,
            status="review",
            applied_steps=steps,
            warnings=warnings,
            metrics=metrics,
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return ProcessingResult(
            image=None,
            status="failed",
            applied_steps=steps,
            warnings=[],
            metrics=metrics,
            elapsed_seconds=time.perf_counter() - started,
            error=str(exc),
        )
