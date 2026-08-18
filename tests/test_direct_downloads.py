import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rembg.product_batch import (
    apply_local_mask_correction,
    cleanup_batch_directory,
    process_product_batch,
)
from rembg.product_image import BatchOptions


class FakeSession:
    def predict(self, image, *args, **kwargs):
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        mask[40:-40, 55:-55] = 255
        return [Image.fromarray(mask, mode="L")]


class BorderTouchingSession:
    def predict(self, image, *args, **kwargs):
        return [Image.new("L", image.size, 255)]


class DirectDownloadTests(unittest.TestCase):
    def setUp(self):
        self.source_directory = tempfile.TemporaryDirectory()
        self.source_path = Path(self.source_directory.name) / "safe.png"
        source = np.full((300, 300, 3), 230, dtype=np.uint8)
        source[40:-40, 55:-55] = (75, 85, 95)
        Image.fromarray(source, mode="RGB").save(self.source_path)
        self.task_directories = []

    def tearDown(self):
        for task_directory in self.task_directories:
            cleanup_batch_directory(task_directory)
        self.source_directory.cleanup()

    def process(self, output_format):
        options = BatchOptions(
            output_format=output_format,
            correct_geometry=False,
            correct_glare=False,
        )
        batch = process_product_batch([self.source_path], options, FakeSession())
        self.task_directories.append(batch.task_directory)
        return batch

    def test_writes_each_supported_output_format(self):
        expected = {
            "jpg": (".jpg", "JPEG"),
            "jpeg": (".jpeg", "JPEG"),
            "png": (".png", "PNG"),
        }

        for output_format, (extension, encoding) in expected.items():
            with self.subTest(output_format=output_format):
                batch = self.process(output_format)
                self.assertEqual(len(batch.output_paths), 1)
                output_path = batch.output_paths[0]
                self.assertEqual(output_path.suffix, extension)
                with Image.open(output_path) as output:
                    self.assertEqual(output.format, encoding)
                    self.assertEqual(output.size, (300, 300))

                manifest = json.loads(
                    (batch.task_directory / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertNotIn("output_size", manifest["options"])
                self.assertNotIn("subject_ratio", manifest["options"])

                self.assertFalse((batch.task_directory / "report.csv").exists())
                self.assertFalse(
                    (batch.task_directory / "safe_white_images.zip").exists()
                )
                self.assertNotIn(batch.task_directory / ".edit", batch.output_paths)

    def test_review_result_uses_review_suffix(self):
        options = BatchOptions(
            output_format="jpeg",
            correct_geometry=False,
            correct_glare=False,
        )
        batch = process_product_batch(
            [self.source_path], options, BorderTouchingSession()
        )
        self.task_directories.append(batch.task_directory)

        self.assertEqual(batch.items[0].result.status, "review")
        self.assertEqual(batch.output_paths[0].name, "safe_review.jpeg")

    def test_local_correction_refreshes_direct_downloads(self):
        batch = self.process("png")
        output_path = batch.output_paths[0]
        before = output_path.read_bytes()
        paint = np.zeros((300, 300), dtype=np.uint8)
        paint[80:140, 80:140] = 255

        _, download_paths = apply_local_mask_correction(
            batch.task_directory, 0, paint, "delete"
        )

        self.assertEqual(download_paths, [output_path])
        self.assertNotEqual(output_path.read_bytes(), before)
        with Image.open(output_path) as output:
            self.assertEqual(output.format, "PNG")
            self.assertEqual(output.size, (300, 300))


if __name__ == "__main__":
    unittest.main()
