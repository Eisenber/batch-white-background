# Original-Size Fidelity White Background Design

## Goal

Improve product detail and edge quality by removing the current square-layout
resizing step. The generated white-background image must retain the oriented
source image's pixel dimensions, aspect ratio, product size, and product
position. A clearly tilted product may be straightened conservatively without
changing the output canvas size or clipping the product.

For example, a 1024 × 768 source produces a 1024 × 768 result. A safe that
occupies a given region in the source remains in that region instead of being
cropped, enlarged, or recentered.

## User-Visible Behavior

- Apply EXIF orientation first, then use the resulting width and height as the
  fixed output dimensions.
- Preserve landscape, portrait, and square aspect ratios exactly.
- Keep the product at its original scale and position when no straightening is
  required.
- Replace only the removed background with pure white.
- Straighten a product only when tilt detection is sufficiently confident and
  the correction is safe within the original-size canvas.
- If a proposed correction could clip the product, skip it and mark the item
  for review instead of shrinking or moving the product.
- Continue offering direct JPG, JPEG, and PNG downloads. CSV and ZIP output do
  not return.

The workbench summary replaces the current `1000 × 1000` and `85%` claims with
copy explaining that output size, product scale, and position follow the
source image.

## Processing Architecture

### Source geometry

The oriented RGB source is the coordinate system for the complete processing
operation. Its `(width, height)` is carried through segmentation, optional
straightening, composition, saved correction assets, and local mask
correction.

The default processing path does not crop the mask bounding box, resize the
product, center it on a new canvas, or enlarge it to a target occupancy ratio.
The existing fixed `output_size` and `subject_ratio` layout settings are
removed from active batch behavior and from newly written manifests.

### Conservative straightening

Straightening corrects roll only. Automatic perspective warping is disabled
for this workflow because it resamples a large part of the product and can
soften text, texture, and narrow edges.

The current line-based angle detector remains the source of roll estimates,
but a rotation is accepted only when all of the following are true:

1. The absolute angle is from 0.6 through 5.0 degrees.
2. Weighted line consistency is at least 0.55.
3. A transformed-mask preflight confirms that the rotated product remains
   at least two pixels inside every edge of the original canvas.

An accepted correction rotates the RGB source and mask together exactly once,
around the product-mask center, onto a canvas with the original dimensions.
The RGB interpolation uses a high-quality image resampler; the mask uses an
alpha-appropriate resampler. Newly exposed pixels are filled with white in the
RGB image and zero in the mask.

If any acceptance condition fails, no geometric transform is applied. A
clipping-risk rejection adds a warning and review status so the user can see
that straightening was intentionally skipped. The processor must not make the
product fit by scaling it down, shifting it toward the center, or expanding
the canvas.

For position checks, the product position is the center of its nonzero mask
bounding box. With no rotation this center and the bounding-box dimensions
must be unchanged. With rotation, the transform must preserve that center to
within one pixel; only the rotated bounding-box dimensions may change.

### Edge refinement and composition

The segmentation mask stays aligned one-to-one with source pixels. Before
composition, the alpha transition is tightened only where the model produced
an excessively wide semi-transparent fringe. The intended visible transition
is approximately one to two pixels at the source resolution.

Refinement must not globally erode the solid product mask, reduce the product
silhouette, or convert the boundary into a hard aliased edge. Fully opaque
interior pixels retain their source RGB values. Composition is a single alpha
blend of the aligned RGB source over pure white; it performs no crop or resize.

When straightening is skipped, the product's fully opaque pixels therefore
pass to the result without geometric interpolation. This is the primary
detail-preservation guarantee for labels, locks, surface texture, and fine
hardware.

Glare correction remains available but is limited to its existing detected
highlight regions. It must not trigger a resize or whole-image geometry pass.

## Batch Data and Local Correction

New batch manifests record the source-aligned image and mask plus the chosen
output format and quality. They no longer rely on a fixed square size or
subject occupancy ratio to reproduce an output.

Local mask correction operates in original-image coordinates. Adding or
removing a painted mask region recomposes the result on the same original-size
canvas, at the same scale and position, and overwrites the selected JPG, JPEG,
or PNG output. The correction path uses the same edge refinement and aligned
composition rules as initial processing.

Compatibility with an old temporary manifest is best-effort: if legacy layout
fields are present, the correction path ignores them and derives canvas size
from the saved aligned source image. Temporary batch data is not a public file
format.

## Status, Metrics, and Errors

Processing metrics retain detected rotation angle and line support. They also
record whether rotation was applied or skipped and, when skipped, the reason.
Perspective displacement is no longer produced by the active workflow.

- Segmentation or serialization failure continues to fail only the affected
  item.
- Empty or unreliable masks retain the existing review/error behavior.
- A tilt with insufficient confidence is left unchanged without claiming that
  straightening occurred.
- A high-confidence tilt that would clip is left unchanged, receives a clear
  clipping-risk warning, and is marked for review.
- No error recovery path may silently resize, crop, or recenter the product.

## Scope

This change includes source-sized composition, position preservation,
conservative roll correction, edge-fringe tightening, local-correction
alignment, UI copy, and regression tests.

It does not add manual rotation controls, perspective correction controls,
background-color choices, super-resolution, generative detail restoration,
automatic cropping, or marketplace-specific output presets.

## Verification

Automated tests cover the following behavior:

- A landscape source produces the same landscape dimensions.
- A portrait source produces the same portrait dimensions.
- A square source retains its exact dimensions.
- With straightening disabled or unnecessary, the mask bounding-box center and
  dimensions remain unchanged within one pixel.
- Fully opaque product pixels remain identical to source pixels when glare
  correction and geometry correction are not applied.
- A clearly tilted synthetic product with strong line support is straightened
  on the original-size canvas without changing scale or position beyond the
  rotation implied by its product-centered transform.
- A near-border product whose proposed rotation would clip is not transformed,
  is marked for review, and reports the clipping-risk reason.
- A synthetic wide alpha fringe becomes narrower while the solid-mask area and
  antialiased boundary remain intact.
- Initial processing and local mask correction both preserve source dimensions
  and alignment.
- JPG, JPEG, and PNG downloads still have matching extensions and encodings,
  and CSV or ZIP files are not created.

The supplied safe image is used for a manual regression check: its 1024 × 768
input must produce a 1024 × 768 result, the safe must remain at its source
location and scale, its detected zero-degree roll must not trigger rotation,
and the lock text and surface detail must remain visibly sharper than in the
previous 1000 × 1000 enlarged result.
