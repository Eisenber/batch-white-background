import base64
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rembg.openai_image import (
    OPENAI_IMAGE_MODEL,
    OPENAI_IMAGE_PROMPT,
    OpenAIAuthenticationError,
    OpenAIContentPolicyError,
    OpenAIConfigurationError,
    OpenAIImageGateway,
    OpenAIImageSession,
    OpenAIServiceError,
    OpenAISubmissionUncertainError,
    choose_output_size,
    _map_sdk_error,
    _resolve_image_config,
)
from rembg.product_batch import process_product_batch
from rembg.product_image import BatchOptions, process_generated_product_image


def encoded_png(width=1536, height=1024, color=(245, 245, 245)):
    output = io.BytesIO()
    Image.new("RGB", (width, height), color).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


class FakeGateway:
    def __init__(self, *, error=None, width=1536, height=1024):
        self.error = error
        self.calls = []
        self.width = width
        self.height = height

    def edit(self, **arguments):
        self.calls.append(arguments)
        if self.error:
            raise self.error
        return base64.b64decode(encoded_png(self.width, self.height))


class IntermittentGateway(FakeGateway):
    def edit(self, **arguments):
        self.calls.append(arguments)
        if len(self.calls) == 2:
            raise OpenAIServiceError("OpenAI 图片服务暂时不可用，未自动重试")
        return base64.b64decode(encoded_png(self.width, self.height))


class FakeImages:
    def __init__(self):
        self.calls = []

    def edit(self, **kwargs):
        self.calls.append(kwargs)
        item = type("ImageItem", (), {"b64_json": encoded_png()})()
        return type("ImageResponse", (), {"data": [item]})()


class FakeClient:
    def __init__(self):
        self.images = FakeImages()


class OpenAIImageTests(unittest.TestCase):
    def test_missing_key_fails_before_gateway_creation(self):
        with self.assertRaisesRegex(OpenAIConfigurationError, "OPENAI_API_KEY"):
            OpenAIImageSession(environ={})

    def test_gateway_forwards_base_url_and_model_to_client(self):
        captured = {}
        client = FakeClient()

        def factory(**kwargs):
            captured.update(kwargs)
            return client

        gateway = OpenAIImageGateway(
            "secret-for-test",
            base_url="https://qlyyds-resource.tech",
            model="gpt-image-2",
            client_factory=factory,
        )
        gateway.edit(
            image_bytes=b"source image",
            filename="source.png",
            content_type="image/png",
            prompt=OPENAI_IMAGE_PROMPT,
            size="1536x1024",
        )

        self.assertEqual(captured["api_key"], "secret-for-test")
        self.assertEqual(captured["base_url"], "https://qlyyds-resource.tech")
        self.assertEqual(client.images.calls[0]["model"], "gpt-image-2")

    def test_resolve_config_prefers_image_vars(self):
        api_key, base_url, model = _resolve_image_config(
            {
                "IMG_API_KEY": "relay-key",
                "IMG_BASE_URL": "https://qlyyds-resource.tech",
                "IMG_MODEL": "gpt-image-2",
            }
        )
        self.assertEqual(api_key, "relay-key")
        self.assertEqual(base_url, "https://qlyyds-resource.tech")
        self.assertEqual(model, "gpt-image-2")

    def test_resolve_config_falls_back_to_legacy_key_and_defaults(self):
        api_key, base_url, model = _resolve_image_config(
            {"OPENAI_API_KEY": "legacy-key"}
        )
        self.assertEqual(api_key, "legacy-key")
        self.assertIsNone(base_url)
        self.assertEqual(model, OPENAI_IMAGE_MODEL)

    def test_gateway_disables_sdk_retries_and_uses_image_edit(self):
        captured = {}
        client = FakeClient()

        def factory(**kwargs):
            captured.update(kwargs)
            return client

        gateway = OpenAIImageGateway("secret-for-test", client_factory=factory)
        source = io.BytesIO(b"source image")
        result = gateway.edit(
            image_bytes=source.getvalue(),
            filename="source.png",
            content_type="image/png",
            prompt=OPENAI_IMAGE_PROMPT,
            size="1536x1024",
        )

        self.assertEqual(captured["api_key"], "secret-for-test")
        self.assertEqual(captured["max_retries"], 0)
        self.assertEqual(len(client.images.calls), 1)
        request = client.images.calls[0]
        self.assertEqual(request["model"], OPENAI_IMAGE_MODEL)
        self.assertEqual(request["quality"], "high")
        self.assertEqual(request["background"], "opaque")
        self.assertEqual(request["output_format"], "png")
        self.assertEqual(request["response_format"], "b64_json")
        self.assertEqual(request["size"], "1536x1024")
        self.assertTrue(result.startswith(b"\x89PNG"))

    def test_prompt_contains_preservation_requirements(self):
        prompt = OPENAI_IMAGE_PROMPT.lower()
        for phrase in (
            "#ffffff",
            "same position",
            "same scale",
            "do not alter any text",
            "number buttons",
            "fingerprint reader",
            "lock hardware",
        ):
            self.assertIn(phrase, prompt)

    def test_content_policy_error_is_distinguished(self):
        error = type(
            "BadRequestError",
            (Exception,),
            {"status_code": 400, "code": "content_policy_violation"},
        )()
        mapped = _map_sdk_error(error)
        self.assertIsInstance(mapped, OpenAIContentPolicyError)
        self.assertIn("内容安全策略", str(mapped))

    def test_legal_size_tracks_source_aspect_ratio(self):
        width, height = choose_output_size(1600, 1200)
        self.assertEqual(width % 16, 0)
        self.assertEqual(height % 16, 0)
        self.assertLessEqual(max(width, height), 3840)
        self.assertGreaterEqual(width * height, 655_360)
        self.assertLessEqual(width * height, 8_294_400)
        self.assertAlmostEqual(width / height, 4 / 3, delta=0.02)

    def test_generated_result_is_restored_to_exact_source_size(self):
        gateway = FakeGateway(width=1536, height=1024)
        session = OpenAIImageSession(
            gateway=gateway,
            environ={"OPENAI_API_KEY": "secret-for-test"},
        )
        source = Image.new("RGB", (641, 479), "gray")

        result = process_generated_product_image(
            source,
            BatchOptions(processing_engine="openai"),
            session,
        )

        self.assertEqual(result.status, "review")
        self.assertEqual(result.image.size, (641, 479))
        self.assertEqual(result.metrics["source_width"], 641.0)
        self.assertEqual(result.metrics["source_height"], 479.0)
        self.assertEqual(result.metrics["returned_width"], 1536.0)
        self.assertEqual(result.metrics["returned_height"], 1024.0)
        self.assertEqual(result.metrics["final_width"], 641.0)
        self.assertEqual(result.metrics["final_height"], 479.0)
        self.assertIn("GPT Image 2", " ".join(result.applied_steps))
        self.assertTrue(any("人工复核" in warning for warning in result.warnings))
        self.assertEqual(len(gateway.calls), 1)

    def test_ambiguous_failure_is_not_retried(self):
        gateway = FakeGateway(
            error=OpenAISubmissionUncertainError(
                "连接中断；请求可能已受理，为避免重复计费未自动重试"
            )
        )
        session = OpenAIImageSession(
            gateway=gateway,
            environ={"OPENAI_API_KEY": "secret-for-test"},
        )

        result = process_generated_product_image(
            Image.new("RGB", (640, 480), "white"),
            BatchOptions(processing_engine="openai"),
            session,
        )

        self.assertEqual(result.status, "failed")
        self.assertIn("未自动重试", result.error)
        self.assertEqual(len(gateway.calls), 1)

    def test_authentication_failure_is_cached_for_the_batch_session(self):
        gateway = FakeGateway(
            error=OpenAIAuthenticationError("OpenAI API Key 无效或没有调用权限")
        )
        session = OpenAIImageSession(
            gateway=gateway,
            environ={"OPENAI_API_KEY": "never-show-this-key"},
        )

        for _ in range(2):
            result = process_generated_product_image(
                Image.new("RGB", (640, 480), "white"),
                BatchOptions(processing_engine="openai"),
                session,
            )
            self.assertEqual(result.status, "failed")
            self.assertNotIn("never-show-this-key", result.error)
        self.assertEqual(len(gateway.calls), 1)

    def test_openai_batch_limit_is_ten_and_local_limit_stays_fifty(self):
        session = OpenAIImageSession(
            gateway=FakeGateway(),
            environ={"OPENAI_API_KEY": "secret-for-test"},
        )
        images = [Image.new("RGB", (640, 480), "white") for _ in range(11)]

        with self.assertRaisesRegex(ValueError, "GPT Image 2.*10"):
            process_product_batch(
                images,
                BatchOptions(processing_engine="openai"),
                session,
            )

    def test_manifest_contains_geometry_but_not_secret(self):
        gateway = FakeGateway()
        session = OpenAIImageSession(
            gateway=gateway,
            environ={"OPENAI_API_KEY": "never-write-this-key"},
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "safe.jpg"
            Image.new("RGB", (640, 480), "gray").save(source)
            batch = process_product_batch(
                [source],
                BatchOptions(processing_engine="openai", output_format="png"),
                session,
            )
            manifest_text = (batch.task_directory / "manifest.json").read_text(
                encoding="utf-8"
            )
            self.assertIn('"source_width": 640.0', manifest_text)
            self.assertIn('"final_width": 640.0', manifest_text)
            self.assertNotIn("never-write-this-key", manifest_text)

    def test_one_service_failure_preserves_other_generated_downloads(self):
        gateway = IntermittentGateway()
        session = OpenAIImageSession(
            gateway=gateway,
            environ={"OPENAI_API_KEY": "secret-for-test"},
        )
        images = [Image.new("RGB", (640, 480), "gray") for _ in range(3)]

        batch = process_product_batch(
            images,
            BatchOptions(processing_engine="openai", output_format="png"),
            session,
        )

        self.assertEqual(batch.counts, {"processed": 0, "review": 2, "failed": 1})
        self.assertEqual(len(batch.output_paths), 2)
        self.assertEqual(len(gateway.calls), 3)

    def test_ui_source_defaults_to_local_and_contains_no_fal_option(self):
        ui_source = (
            Path(__file__).parents[1] / "rembg" / "product_ui.py"
        ).read_text(encoding="utf-8")
        self.assertIn('(\"本地离线 · BiRefNet\", \"local\")', ui_source)
        self.assertIn('(\"高质量生成 · GPT Image 2\", \"openai\")', ui_source)
        self.assertIn('value="local"', ui_source)
        self.assertIn("生成式编辑", ui_source)
        self.assertNotIn("fal.ai", ui_source)


if __name__ == "__main__":
    unittest.main()
