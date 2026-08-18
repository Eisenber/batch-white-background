"""Single-submit OpenAI GPT Image 2 editing for product photography."""

from __future__ import annotations

import base64
import binascii
import io
import math
import os
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from PIL import Image
from PIL.Image import Image as PILImage

from .product_image import GeneratedProductImage


OPENAI_IMAGE_MODEL = "gpt-image-2"
OPENAI_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
OPENAI_MAX_RESULT_BYTES = 100 * 1024 * 1024
OPENAI_MIN_PIXELS = 655_360
OPENAI_MAX_PIXELS = 8_294_400
OPENAI_MAX_EDGE = 3840

OPENAI_IMAGE_PROMPT = """Replace only the surrounding environment and supporting
surface with a clean, opaque pure white #FFFFFF background. Keep the safe in
exactly the same position, at exactly the same scale, with the same proportions,
orientation, perspective, and framing as the source. Do not crop or move the
safe. Do not alter any text, logos, number buttons, fingerprint reader,
lock hardware, handles, seams, edges, colors, reflections, surface texture, or
construction details. Do not add or remove any product component. Preserve the
source product faithfully. A very subtle natural contact shadow is allowed only
where needed to keep the product grounded. Produce a professional e-commerce
product photograph with no decorative elements."""


class OpenAIImageError(RuntimeError):
    """Safe user-facing OpenAI image error."""


class OpenAIConfigurationError(OpenAIImageError):
    pass


class OpenAIAuthenticationError(OpenAIImageError):
    pass


class OpenAIQuotaError(OpenAIImageError):
    pass


class OpenAIRateLimitError(OpenAIImageError):
    pass


class OpenAIInputError(OpenAIImageError):
    pass


class OpenAIContentPolicyError(OpenAIImageError):
    pass


class OpenAISubmissionUncertainError(OpenAIImageError):
    pass


class OpenAIServiceError(OpenAIImageError):
    pass


class OpenAIImageGatewayProtocol(Protocol):
    def edit(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        content_type: str,
        prompt: str,
        size: str,
    ) -> bytes: ...


def choose_output_size(source_width: int, source_height: int) -> tuple[int, int]:
    """Choose the legal GPT size nearest to source ratio, then source area."""

    if source_width <= 0 or source_height <= 0:
        raise ValueError("图片尺寸无效")
    source_ratio = source_width / source_height
    target_ratio = min(3.0, max(1.0 / 3.0, source_ratio))
    target_area = min(
        OPENAI_MAX_PIXELS,
        max(OPENAI_MIN_PIXELS, source_width * source_height),
    )
    candidates: list[tuple[tuple[float, float], int, int]] = []
    for width in range(16, OPENAI_MAX_EDGE + 1, 16):
        raw_height = width / target_ratio
        height = max(16, int(round(raw_height / 16.0)) * 16)
        if height > OPENAI_MAX_EDGE:
            continue
        pixels = width * height
        ratio = width / height
        if not OPENAI_MIN_PIXELS <= pixels <= OPENAI_MAX_PIXELS:
            continue
        if not 1.0 / 3.0 <= ratio <= 3.0:
            continue
        score = (
            abs(math.log(ratio / target_ratio)),
            abs(math.log(pixels / target_area)),
        )
        candidates.append((score, width, height))
    if not candidates:
        raise OpenAIInputError("无法为这张图片选择合法的 GPT 输出尺寸")
    _, width, height = min(candidates)
    return width, height


def _prepare_upload(image: PILImage) -> tuple[bytes, str, str]:
    png = io.BytesIO()
    image.convert("RGB").save(png, format="PNG", optimize=True)
    if png.tell() < OPENAI_MAX_UPLOAD_BYTES:
        return png.getvalue(), "source.png", "image/png"

    for quality in (95, 90, 85, 80):
        jpeg = io.BytesIO()
        image.convert("RGB").save(
            jpeg,
            format="JPEG",
            quality=quality,
            subsampling=0,
            optimize=True,
        )
        if jpeg.tell() < OPENAI_MAX_UPLOAD_BYTES:
            return jpeg.getvalue(), "source.jpg", "image/jpeg"
    raise OpenAIInputError("原图文件过大，压缩后仍超过 OpenAI 50MB 上传限制")


def _exception_code(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    if code:
        return str(code).lower()
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        nested = body.get("error", body)
        if isinstance(nested, dict) and nested.get("code"):
            return str(nested["code"]).lower()
    return ""


def _map_sdk_error(exc: Exception) -> OpenAIImageError:
    """Map SDK exceptions without exposing request bodies or credentials."""

    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    code = _exception_code(exc)
    if name == "AuthenticationError" or status == 401:
        return OpenAIAuthenticationError("OpenAI API Key 无效或已被撤销")
    if name == "PermissionDeniedError" or status == 403:
        return OpenAIAuthenticationError("OpenAI API Key 没有 GPT Image 2 调用权限")
    if "quota" in code or "billing" in code or "balance" in code:
        return OpenAIQuotaError("OpenAI API 额度不足或账户计费不可用")
    if "content_policy" in code or "safety" in code:
        return OpenAIContentPolicyError("图片被 OpenAI 内容安全策略拒绝")
    if name == "RateLimitError" or status == 429:
        return OpenAIRateLimitError("OpenAI 请求频率过高，请稍后手动重试")
    if name in {"APIConnectionError", "APITimeoutError"}:
        return OpenAISubmissionUncertainError(
            "连接 OpenAI 时中断；请求可能已受理，为避免重复计费未自动重试"
        )
    if name == "BadRequestError" or status in {400, 413, 415, 422}:
        return OpenAIInputError("OpenAI 拒绝了图片或编辑参数，请检查图片格式和内容")
    if isinstance(status, int) and status >= 500:
        return OpenAIServiceError("OpenAI 图片服务暂时不可用，未自动重试")
    return OpenAIServiceError("OpenAI 图片编辑失败，未自动重试")


class OpenAIImageGateway:
    """Official SDK boundary configured to submit every edit at most once."""

    def __init__(
        self,
        api_key: str,
        *,
        client_factory: Callable[..., Any] | None = None,
        timeout_seconds: float = 180.0,
    ):
        if client_factory is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise OpenAIConfigurationError(
                    "缺少 OpenAI Python 依赖，请先安装 requirements-openai.txt"
                ) from exc
            client_factory = OpenAI
        self._client = client_factory(
            api_key=api_key,
            max_retries=0,
            timeout=timeout_seconds,
        )

    def edit(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        content_type: str,
        prompt: str,
        size: str,
    ) -> bytes:
        try:
            response = self._client.images.edit(
                model=OPENAI_IMAGE_MODEL,
                image=(filename, image_bytes, content_type),
                prompt=prompt,
                size=size,
                quality="high",
                background="opaque",
                output_format="png",
                response_format="b64_json",
            )
        except Exception as exc:
            raise _map_sdk_error(exc) from exc

        data = getattr(response, "data", None)
        if not isinstance(data, list) or not data:
            raise OpenAIServiceError("OpenAI 响应中没有生成图片")
        encoded = getattr(data[0], "b64_json", None)
        if encoded is None and isinstance(data[0], dict):
            encoded = data[0].get("b64_json")
        if not isinstance(encoded, str) or not encoded:
            raise OpenAIServiceError("OpenAI 响应缺少图片数据")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise OpenAIServiceError("OpenAI 返回的图片编码无效") from exc
        if not raw or len(raw) > OPENAI_MAX_RESULT_BYTES:
            raise OpenAIServiceError("OpenAI 返回的图片大小异常")
        return raw


class OpenAIImageSession:
    """Prepare and generate one complete GPT Image 2 product photograph."""

    def __init__(
        self,
        *,
        gateway: OpenAIImageGatewayProtocol | None = None,
        environ: Mapping[str, str] | None = None,
        stage_callback: Callable[[str], None] | None = None,
    ):
        environment = os.environ if environ is None else environ
        api_key = environment.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise OpenAIConfigurationError(
                "未配置 OPENAI_API_KEY；请在启动工作台的终端设置后重新运行"
            )
        self._gateway = gateway or OpenAIImageGateway(api_key)
        self._stage_callback = stage_callback
        self._fatal_error: OpenAIImageError | None = None

    def _stage(self, message: str) -> None:
        if self._stage_callback:
            self._stage_callback(message)

    def generate(self, image: PILImage) -> GeneratedProductImage:
        if self._fatal_error is not None:
            raise self._fatal_error
        self._stage("正在准备原图")
        width, height = choose_output_size(image.width, image.height)
        image_bytes, filename, content_type = _prepare_upload(image)
        self._stage("正在上传至 OpenAI")
        self._stage("GPT Image 2 正在生成白底图")
        try:
            raw = self._gateway.edit(
                image_bytes=image_bytes,
                filename=filename,
                content_type=content_type,
                prompt=OPENAI_IMAGE_PROMPT,
                size=f"{width}x{height}",
            )
        except (OpenAIAuthenticationError, OpenAIQuotaError) as exc:
            self._fatal_error = exc
            raise
        try:
            output = Image.open(io.BytesIO(raw))
            output.load()
            output = output.convert("RGB")
        except (OSError, ValueError) as exc:
            raise OpenAIServiceError("OpenAI 返回的图片无法解码") from exc
        self._stage("正在恢复原图尺寸")
        return GeneratedProductImage(
            image=output,
            model=OPENAI_IMAGE_MODEL,
            requested_width=width,
            requested_height=height,
            returned_width=output.width,
            returned_height=output.height,
        )
