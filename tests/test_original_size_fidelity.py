import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from rembg.product_image import (
    BatchOptions,
    _refine_mask_edge,
    _rotate_pair,
    _rotation_angle,
    compose_white_canvas,
    process_product_image,
)


class MaskSession:
    def __init__(self, mask: np.ndarray):
        self.mask = mask

    def predict(self, image, *args, **kwargs):
        return [Image.fromarray(self.mask, mode="L")]


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask >= 12)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


class OriginalSizeFidelityTests(unittest.TestCase):
    def test_composition_keeps_dimensions_position_and_opaque_pixels(self):
        for width, height in ((320, 180), (180, 320), (240, 240)):
            with self.subTest(size=(width, height)):
                source = np.zeros((height, width, 3), dtype=np.uint8)
                source[:, :] = (25, 45, 65)
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[31 : height - 27, 43 : width - 39] = 255

                output, metrics = compose_white_canvas(source, mask)
                result = np.asarray(output)

                self.assertEqual(output.size, (width, height))
                self.assertTrue(np.array_equal(result[mask == 255], source[mask == 255]))
                self.assertTrue(np.all(result[mask == 0] == 255))
                self.assertEqual(metrics["output_subject_scale"], 1.0)
                self.assertEqual(
                    metrics["output_subject_center_x"],
                    (43 + width - 40) / 2.0,
                )
                self.assertEqual(
                    metrics["output_subject_center_y"],
                    (31 + height - 28) / 2.0,
                )

    def test_wide_soft_fringe_is_tightened_without_losing_solid_core(self):
        binary = np.zeros((140, 180), dtype=np.uint8)
        binary[35:105, 45:135] = 255
        soft = cv2.GaussianBlur(binary, (0, 0), sigmaX=5.0)
        before_soft_pixels = int(((soft > 0) & (soft < 255)).sum())
        before_solid = soft == 255

        refined, estimated_width, changed = _refine_mask_edge(soft)

        self.assertTrue(changed)
        self.assertGreater(estimated_width, 2.0)
        self.assertLess(int(((refined > 0) & (refined < 255)).sum()), before_soft_pixels)
        self.assertTrue(np.all(refined[before_solid] == 255))
        self.assertTrue(((refined > 0) & (refined < 255)).any())

    def test_safe_rotation_keeps_original_canvas_and_subject_center(self):
        source = np.full((220, 320, 3), 90, dtype=np.uint8)
        mask = np.zeros((220, 320), dtype=np.uint8)
        mask[65:165, 90:250] = 255
        before = mask_bbox(mask)
        before_center = ((before[0] + before[2]) / 2, (before[1] + before[3]) / 2)

        rotated_source, rotated_mask, applied = _rotate_pair(source, mask, 3.0)
        after = mask_bbox(rotated_mask)
        after_center = ((after[0] + after[2]) / 2, (after[1] + after[3]) / 2)

        self.assertTrue(applied)
        self.assertEqual(rotated_source.shape, source.shape)
        self.assertEqual(rotated_mask.shape, mask.shape)
        self.assertLessEqual(abs(after_center[0] - before_center[0]), 1.0)
        self.assertLessEqual(abs(after_center[1] - before_center[1]), 1.0)

    @patch("rembg.product_image.cv2.HoughLinesP")
    def test_rotation_angle_accepts_flat_opencv5_hough_lines(self, hough_lines):
        image = np.full((200, 300, 3), 128, dtype=np.uint8)
        mask = np.zeros((200, 300), dtype=np.uint8)
        mask[50:150, 70:230] = 255
        # OpenCV 5 returns HoughLinesP as (N, 4); OpenCV 4 wrapped them as
        # (N, 1, 4).  Four parallel ~2.86° lines must still be read correctly.
        hough_lines.return_value = np.array(
            [
                [0, 0, 200, 10],
                [0, 10, 200, 20],
                [0, 20, 200, 30],
                [0, 30, 200, 40],
            ],
            dtype=np.int32,
        )

        angle, support = _rotation_angle(image, mask)

        self.assertAlmostEqual(angle, 2.86, places=2)
        self.assertGreaterEqual(support, 0.55)

    def test_near_border_rotation_is_skipped_instead_of_clipped(self):
        source = np.full((180, 260, 3), 90, dtype=np.uint8)
        mask = np.zeros((180, 260), dtype=np.uint8)
        mask[20:160, 1:170] = 255

        rotated_source, rotated_mask, applied = _rotate_pair(source, mask, 5.0)

        self.assertFalse(applied)
        self.assertIs(rotated_source, source)
        self.assertIs(rotated_mask, mask)

    def test_processing_preserves_source_geometry_when_rotation_not_needed(self):
        width, height = 360, 240
        source = np.full((height, width, 3), 238, dtype=np.uint8)
        source[55:195, 70:285] = (61, 73, 89)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[55:195, 70:285] = 255

        result = process_product_image(
            Image.fromarray(source, mode="RGB"),
            BatchOptions(correct_geometry=False, correct_glare=False),
            MaskSession(mask),
        )

        self.assertIsNotNone(result.image)
        self.assertEqual(result.image.size, (width, height))
        self.assertEqual(result.metrics["rotation_status"], "disabled")
        self.assertEqual(result.metrics["output_subject_scale"], 1.0)
        output = np.asarray(result.image)
        self.assertTrue(np.array_equal(output[mask == 255], source[mask == 255]))

    @patch("rembg.product_image._rotation_angle", return_value=(3.0, 0.9))
    def test_processing_marks_safe_rotation_as_applied(self, _rotation_angle):
        source = np.full((220, 320, 3), 100, dtype=np.uint8)
        mask = np.zeros((220, 320), dtype=np.uint8)
        mask[65:165, 90:250] = 255

        result = process_product_image(
            Image.fromarray(source, mode="RGB"),
            BatchOptions(correct_glare=False),
            MaskSession(mask),
        )

        self.assertEqual(result.image.size, (320, 220))
        self.assertEqual(result.metrics["rotation_status"], "applied")
        self.assertTrue(any("主体原位回正" in step for step in result.applied_steps))

    @patch("rembg.product_image._rotation_angle", return_value=(5.0, 0.9))
    def test_processing_flags_rotation_that_would_clip(self, _rotation_angle):
        source = np.full((180, 260, 3), 100, dtype=np.uint8)
        mask = np.zeros((180, 260), dtype=np.uint8)
        mask[20:160, 1:170] = 255

        result = process_product_image(
            Image.fromarray(source, mode="RGB"),
            BatchOptions(correct_glare=False),
            MaskSession(mask),
        )

        self.assertEqual(result.image.size, (260, 180))
        self.assertEqual(result.status, "review")
        self.assertEqual(result.metrics["rotation_status"], "skipped_clipping_risk")
        self.assertTrue(any("避免裁切" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()
