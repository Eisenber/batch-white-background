import base64
import io
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image

from rembg.fal_session import (
    FAL_APPLICATION,
    FAL_MODEL,
    FAL_RESOLUTION,
    FalAuthenticationError,
    FalBiRefNetSession,
    FalConfigurationError,
    FalJob,
    FalSubmissionUncertainError,
    HttpFalGateway,
)
from rembg.product_image import BatchOptions, process_product_image


def mask_data_uri(width=160, height=120):
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[20:-20, 25:-25] = 255
    output = io.BytesIO()
    Image.fromarray(mask, mode="L").save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


class FakeFalGateway:
    def __init__(self, result=None, submit_error=None):
        self.result_payload = result or {"image": {"url": mask_data_uri()}}
        self.submit_error = submit_error
        self.submit_calls = 0
        self.arguments = None

    def submit(self, arguments):
        self.submit_calls += 1
        self.arguments = arguments
        if self.submit_error:
            raise self.submit_error
        return FalJob(
            "request-123456",
            "https://queue.fal.run/status",
            "https://queue.fal.run/result",
        )

    def status(self, job):
        return {"status": "COMPLETED"}

    def result(self, job):
        return self.result_payload


class FalCloudTests(unittest.TestCase):
    def test_missing_key_fails_before_gateway_creation(self):
        with self.assertRaisesRegex(FalConfigurationError, "FAL_KEY"):
            FalBiRefNetSession(environ={})

    def test_fixed_2k_mask_only_parameters(self):
        gateway = FakeFalGateway()
        session = FalBiRefNetSession(
            gateway=gateway, environ={"FAL_KEY": "secret-for-test"}
        )

        masks = session.predict(Image.new("RGB", (320, 240), "gray"))

        self.assertEqual(len(masks), 1)
        self.assertEqual(gateway.arguments["model"], FAL_MODEL)
        self.assertEqual(gateway.arguments["operating_resolution"], FAL_RESOLUTION)
        self.assertTrue(gateway.arguments["output_mask"])
        self.assertTrue(gateway.arguments["mask_only"])
        self.assertTrue(gateway.arguments["sync_mode"])
        self.assertEqual(gateway.arguments["output_format"], "png")
        self.assertTrue(gateway.arguments["image_url"].startswith("data:image/png;base64,"))

    def test_cloud_mask_is_aligned_to_source_before_composition(self):
        gateway = FakeFalGateway()
        session = FalBiRefNetSession(
            gateway=gateway, environ={"FAL_KEY": "secret-for-test"}
        )
        source = np.full((240, 320, 3), 230, dtype=np.uint8)
        source[40:200, 50:270] = (70, 80, 90)

        result = process_product_image(
            Image.fromarray(source, mode="RGB"),
            BatchOptions(
                processing_engine="fal",
                correct_geometry=False,
                correct_glare=False,
            ),
            session,
        )

        self.assertIsNotNone(result.image)
        self.assertEqual(result.image.size, (320, 240))
        self.assertEqual(result.metrics["mask_returned_width"], 160.0)
        self.assertEqual(result.metrics["mask_returned_height"], 120.0)
        self.assertEqual(result.metrics["mask_resized_to_source"], 1.0)
        self.assertEqual(result.metrics["output_subject_scale"], 1.0)
        self.assertIn("fal.ai BiRefNet Light 2K", result.applied_steps[1])

    def test_authentication_failure_is_cached_without_more_submissions(self):
        error = FalAuthenticationError("fal.ai API Key 无效或没有调用权限")
        gateway = FakeFalGateway(submit_error=error)
        session = FalBiRefNetSession(
            gateway=gateway, environ={"FAL_KEY": "never-show-this-key"}
        )

        for _ in range(2):
            with self.assertRaises(FalAuthenticationError) as context:
                session.predict(Image.new("RGB", (32, 32), "white"))
            self.assertNotIn("never-show-this-key", str(context.exception))
        self.assertEqual(gateway.submit_calls, 1)

    @patch("rembg.fal_session.time.sleep")
    def test_rate_limit_retries_are_bounded(self, _sleep):
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) < 3:
                return httpx.Response(429, json={"detail": "limited"})
            return httpx.Response(
                200,
                json={
                    "request_id": "request-123456",
                    "status_url": "https://queue.fal.run/status",
                    "response_url": "https://queue.fal.run/result",
                },
            )

        gateway = HttpFalGateway("secret-for-test")
        gateway._client = httpx.Client(transport=httpx.MockTransport(handler))

        job = gateway.submit({"image_url": "data:image/png;base64,AA=="})

        self.assertEqual(job.request_id, "request-123456")
        self.assertEqual(len(calls), 3)
        self.assertEqual(_sleep.call_count, 2)

    def test_ambiguous_submission_failure_is_not_retried(self):
        calls = []

        def handler(request):
            calls.append(request)
            raise httpx.ConnectError("connection lost", request=request)

        gateway = HttpFalGateway("secret-for-test")
        gateway._client = httpx.Client(transport=httpx.MockTransport(handler))

        with self.assertRaisesRegex(FalSubmissionUncertainError, "未自动重试"):
            gateway.submit({"image_url": "data:image/png;base64,AA=="})
        self.assertEqual(len(calls), 1)

    def test_application_name_is_stable(self):
        self.assertEqual(FAL_APPLICATION, "fal-ai/birefnet/v2")


if __name__ == "__main__":
    unittest.main()
