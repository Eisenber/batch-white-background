"""Cost-safe fal.ai BiRefNet mask inference for the product workbench."""

from __future__ import annotations

import base64
import binascii
import io
import os
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol
from urllib.parse import urlparse

import httpx
from PIL import Image
from PIL.Image import Image as PILImage


FAL_APPLICATION = "fal-ai/birefnet/v2"
FAL_QUEUE_URL = f"https://queue.fal.run/{FAL_APPLICATION}"
FAL_MODEL = "General Use (Light 2K)"
FAL_RESOLUTION = "2048x2048"
MAX_MASK_BYTES = 24 * 1024 * 1024


class FalSessionError(RuntimeError):
    """A safe, user-facing cloud inference error."""


class FalConfigurationError(FalSessionError):
    pass


class FalAuthenticationError(FalSessionError):
    pass


class FalSubmissionUncertainError(FalSessionError):
    pass


class FalRateLimitError(FalSessionError):
    pass


class FalRemoteError(FalSessionError):
    pass


class FalTransientError(FalSessionError):
    pass


@dataclass(frozen=True)
class FalJob:
    request_id: str
    status_url: str
    response_url: str


class FalGateway(Protocol):
    def submit(self, arguments: dict) -> FalJob: ...

    def status(self, job: FalJob) -> dict: ...

    def result(self, job: FalJob) -> dict: ...


def _fal_url(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise FalRemoteError(f"fal.ai 响应缺少 {field}")
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname not in {
        "queue.fal.run",
        "fal.run",
    }:
        raise FalRemoteError(f"fal.ai 返回了无效的 {field}")
    return value


class HttpFalGateway:
    """Minimal fal queue client with no ambiguous submission retries."""

    def __init__(self, api_key: str, timeout: float = 30.0):
        self._headers = {
            "Authorization": f"Key {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=timeout)

    def _response_error(self, response: httpx.Response) -> FalSessionError:
        if response.status_code in (401, 403):
            return FalAuthenticationError("fal.ai API Key 无效或没有调用权限")
        if response.status_code == 429:
            return FalRateLimitError("fal.ai 请求过于频繁，请稍后重试")
        return FalRemoteError(f"fal.ai 服务返回 HTTP {response.status_code}")

    @staticmethod
    def _json(response: httpx.Response) -> dict:
        try:
            payload = response.json()
        except ValueError as exc:
            raise FalRemoteError("fal.ai 返回了无法解析的响应") from exc
        if not isinstance(payload, dict):
            raise FalRemoteError("fal.ai 返回了无效的响应结构")
        return payload

    def submit(self, arguments: dict) -> FalJob:
        for attempt in range(3):
            try:
                response = self._client.post(
                    FAL_QUEUE_URL, headers=self._headers, json=arguments
                )
            except httpx.RequestError as exc:
                raise FalSubmissionUncertainError(
                    "提交 fal.ai 时连接中断；任务可能已受理，为避免重复计费未自动重试"
                ) from exc
            if response.status_code == 429 and attempt < 2:
                time.sleep(2**attempt)
                continue
            if response.is_error:
                raise self._response_error(response)
            payload = self._json(response)
            request_id = payload.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise FalRemoteError("fal.ai 未返回任务编号")
            return FalJob(
                request_id=request_id,
                status_url=_fal_url(payload.get("status_url"), "status_url"),
                response_url=_fal_url(payload.get("response_url"), "response_url"),
            )
        raise FalRateLimitError("fal.ai 请求过于频繁，请稍后重试")

    def _get(self, url: str) -> dict:
        try:
            response = self._client.get(url, headers=self._headers)
        except httpx.RequestError as exc:
            raise FalTransientError("查询 fal.ai 任务状态时网络中断") from exc
        if response.is_error:
            raise self._response_error(response)
        return self._json(response)

    def status(self, job: FalJob) -> dict:
        return self._get(job.status_url)

    def result(self, job: FalJob) -> dict:
        return self._get(job.response_url)


def _encode_source(image: PILImage) -> str:
    output = io.BytesIO()
    image.convert("RGB").save(output, format="PNG", optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _decode_mask_data_uri(value: object) -> PILImage:
    if not isinstance(value, str) or not value.startswith("data:image/png;base64,"):
        raise FalRemoteError("fal.ai 未按同步模式返回 PNG 蒙版数据")
    encoded = value.partition(",")[2]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FalRemoteError("fal.ai 返回的蒙版编码无效") from exc
    if not raw or len(raw) > MAX_MASK_BYTES:
        raise FalRemoteError("fal.ai 返回的蒙版大小异常")
    try:
        mask = Image.open(io.BytesIO(raw))
        mask.load()
        return mask.convert("L")
    except (OSError, ValueError) as exc:
        raise FalRemoteError("fal.ai 返回的蒙版无法解码") from exc


class FalBiRefNetSession:
    """Expose fal.ai mask inference through the existing session boundary."""

    def __init__(
        self,
        *,
        gateway: FalGateway | None = None,
        environ: Mapping[str, str] | None = None,
        stage_callback: Callable[[str], None] | None = None,
        timeout_seconds: float = 120.0,
        poll_interval: float = 0.5,
    ):
        environment = os.environ if environ is None else environ
        api_key = environment.get("FAL_KEY", "").strip()
        if not api_key:
            raise FalConfigurationError(
                "未配置 FAL_KEY；请在启动工作台的终端设置后重新运行"
            )
        self._gateway = gateway or HttpFalGateway(api_key)
        self._stage_callback = stage_callback
        self._timeout_seconds = timeout_seconds
        self._poll_interval = poll_interval
        self._fatal_error: FalAuthenticationError | None = None

    def _stage(self, message: str) -> None:
        if self._stage_callback:
            self._stage_callback(message)

    @staticmethod
    def _arguments(image: PILImage) -> dict:
        return {
            "model": FAL_MODEL,
            "operating_resolution": FAL_RESOLUTION,
            "output_mask": True,
            "mask_only": True,
            "sync_mode": True,
            "output_format": "png",
            "image_url": _encode_source(image),
        }

    def predict(self, image: PILImage, *args, **kwargs) -> list[PILImage]:
        if self._fatal_error is not None:
            raise self._fatal_error
        self._stage("正在上传原图并提交 fal.ai")
        try:
            job = self._gateway.submit(self._arguments(image))
        except FalAuthenticationError as exc:
            self._fatal_error = exc
            raise

        self._stage("fal.ai 2K 蒙版推理中")
        deadline = time.monotonic() + self._timeout_seconds
        while time.monotonic() < deadline:
            try:
                payload = self._gateway.status(job)
            except FalAuthenticationError as exc:
                self._fatal_error = exc
                raise
            except FalTransientError:
                time.sleep(self._poll_interval)
                continue
            status = str(payload.get("status", "")).upper()
            if status == "COMPLETED":
                break
            if status in {"FAILED", "CANCELLED"}:
                raise FalRemoteError("fal.ai 蒙版任务处理失败")
            time.sleep(self._poll_interval)
        else:
            suffix = job.request_id[-6:] if job.request_id else "unknown"
            raise FalRemoteError(
                f"fal.ai 任务等待超时（任务号末尾 …{suffix}），未重复提交"
            )

        self._stage("正在下载并校验 fal.ai 蒙版")
        result_deadline = time.monotonic() + min(15.0, self._timeout_seconds)
        while True:
            try:
                payload = self._gateway.result(job)
                break
            except FalAuthenticationError as exc:
                self._fatal_error = exc
                raise
            except FalTransientError:
                if time.monotonic() >= result_deadline:
                    raise FalRemoteError("下载 fal.ai 蒙版超时，未重复提交")
                time.sleep(self._poll_interval)
        image_payload = payload.get("image")
        if not isinstance(image_payload, dict):
            raise FalRemoteError("fal.ai 响应缺少蒙版图片")
        mask = _decode_mask_data_uri(image_payload.get("url"))
        self._stage("云端蒙版完成，正在本地合成白底图")
        return [mask]
