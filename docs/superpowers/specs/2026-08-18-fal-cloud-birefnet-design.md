# fal.ai Cloud BiRefNet Design

## Goal

Add an optional cloud processing engine that obtains a high-resolution product
mask from fal.ai's `fal-ai/birefnet/v2` endpoint while keeping all composition,
geometry checks, corrections, and file encoding local. The default cloud
configuration targets a typical per-image duration in the tens of seconds
without requiring a subscription or significant local inference compute.

The product pixels must never be regenerated. The cloud service supplies only
an alpha mask; the existing source-sized, position-preserving pipeline uses the
oriented source RGB pixels to produce the final white-background image.

## User Experience

The workbench adds a processing-engine control with two choices:

- `云端高精度 · fal.ai 2K` is selected by default.
- `本地离线 · BiRefNet` retains the current local inference path as a manual
  fallback.

The cloud choice displays a short notice that the source image is sent to
fal.ai and that successful calls consume prepaid usage credits. It does not
display or accept an API key in the browser.

Progress text distinguishes four stages: preparing/uploading the image,
waiting for cloud inference, downloading the mask, and composing locally. The
existing result table, review state, local mask editor, and direct JPG, JPEG,
and PNG downloads remain available.

There is no automatic fallback from cloud to local processing. A fallback must
be chosen explicitly so the user always knows which engine produced a mask.

## Cloud Model Configuration

The first version uses these fixed fal.ai parameters:

- Endpoint: `fal-ai/birefnet/v2`
- Model: `General Use (Light 2K)`
- Operating resolution: `2048x2048`
- Mask-only response: enabled
- Synchronous privacy mode: enabled
- Output format: PNG

`General Use (Light 2K)` is selected to balance 2K edge fidelity with the
requested processing time. The UI does not expose the other fal.ai variants in
this version. Adding a slower Heavy or Matting choice is outside this scope and
can be evaluated later using real difficult-image results.

Mask-only mode intentionally skips fal.ai foreground refinement. The local
pipeline owns edge-alpha refinement and composition, which prevents a cloud
foreground render from replacing source pixels.

## Architecture

### Engine selection

`BatchOptions` gains a validated processing-engine value such as `fal` or
`local`. The UI session provider creates the existing rembg session for local
mode and a new `FalBiRefNetSession` for cloud mode. Both expose the existing
`predict(image) -> list[mask]` boundary, so product cleanup and composition do
not depend on the mask provider.

### fal.ai gateway

`FalBiRefNetSession` is a small adapter responsible only for:

1. Reading `FAL_KEY` from the server process environment.
2. Encoding the oriented source as a lossless PNG data URI.
3. Submitting one queue request with the fixed model parameters.
4. Polling the accepted request by its request ID.
5. Downloading and validating the returned grayscale PNG mask.
6. Returning a Pillow `L` image aligned to the source dimensions.

The adapter uses fal.ai's documented HTTPS queue protocol through the existing
`httpx` dependency, behind a narrow internal gateway that tests can replace
without contacting fal.ai. It does not use the official `fal-client` submit
helper because that helper automatically retries ambiguous transport failures,
which could duplicate a billable submission. The API key is never accepted as
a function parameter passed from the browser and is never included in
exceptions, logs, manifests, or result data.

The source is sent as an in-request data URI instead of first uploading it to a
separate public URL. `sync_mode` is enabled so returned media is represented as
a data URI and is not retained in fal.ai request history. This minimizes data
persistence but does not change the fact that fal.ai receives the source for
inference.

### Local composition

After the mask is received, the existing pipeline continues unchanged:

- Normalize mask dimensions to the oriented source dimensions if the endpoint
  response differs.
- Retain the strongest connected foreground and run quality checks.
- Apply conservative safe straightening only when requested and safe.
- Tighten an overly broad semi-transparent fringe locally.
- Composite the original aligned RGB pixels over pure white without cropping,
  resizing, recentering, or generative reconstruction.

The cloud engine does not perform glare correction, geometry reconstruction,
or image generation. Those remain explicitly controlled local operations.

## Data Flow

1. The user uploads 1–50 PNG, JPG, or JPEG files and chooses the engine.
2. The server validates that `FAL_KEY` exists before creating a paid request.
3. For each cloud item, the server decodes EXIF orientation and sRGB colour.
4. The adapter submits the lossless source data and receives a request ID.
5. The adapter polls that request ID until completion or a 120-second timeout.
6. The adapter retrieves and validates the returned mask.
7. The normal product pipeline composes the white-background output locally.
8. The batch writes the chosen JPG, JPEG, or PNG format and exposes it directly
   for download.
9. Local painted-mask correction reuses the stored aligned source and mask and
   does not call fal.ai again.

Batch processing remains sequential. This limits accidental spend, avoids a
burst of cloud jobs, and preserves the existing per-item progress semantics.

## Authentication and Cost Safety

`FAL_KEY` is the only accepted credential source. If it is absent, cloud mode
stops before processing and displays setup guidance. The key is not stored in
the repository, `.env` files created by the application, temporary batch
directories, or browser state.

The gateway treats the queue request ID as the idempotency boundary:

- After an ID is received, every status check and result retrieval uses that
  same ID; the job is never resubmitted.
- A definite pre-acceptance rate-limit response may be retried at most twice
  with bounded exponential backoff.
- An ambiguous connection failure during submission is not automatically
  retried because the service may already have accepted a billable job.
- Server failures, authentication errors, and malformed responses do not
  trigger a different paid provider.

No price estimate is hard-coded because fal.ai model prices may change. The UI
states that cloud mode consumes prepaid credits and links users to the current
fal.ai pricing page.

## Error Handling

- Missing `FAL_KEY`: fail before submitting any item with a clear setup error.
- HTTP 401/403: report invalid or unauthorized credentials without echoing the
  response headers or key.
- HTTP 429 before acceptance: retry at most twice, then fail that item.
- Queue timeout after acceptance: mark the item failed and retain the request
  ID only in a redacted diagnostic message; do not submit a replacement.
- Network failure after acceptance: report that the remote job status is
  unknown and do not duplicate it.
- Non-image or undecodable response: fail the item.
- Empty, nearly full, or border-contact mask: use the existing quality and
  review rules.
- Mask-size mismatch: resize the mask once to the oriented source dimensions
  using alpha-appropriate interpolation and record the returned dimensions.

One failed item does not stop the rest of the batch. A cloud-wide credential
failure discovered on the first request stops the remaining items to avoid
repeating guaranteed failures.

## Performance Expectations

The target is a typical duration in the tens of seconds per image on a normal
network and uncongested queue. It is a target, not an SLA: upload bandwidth and
fal.ai queue latency can extend the total time.

Local work is limited to image decoding, PNG encoding, mask validation,
quality checks, optional affine rotation, alpha composition, and final file
encoding. No local PyTorch installation or GPU inference is required for cloud
mode.

## Compatibility and Scope

The implementation retains:

- Original source dimensions, aspect ratio, product scale, and position.
- Conservative roll correction and clipping-risk review behavior.
- Local mask editing without another paid call.
- Direct JPG, JPEG, and PNG downloads with no CSV or ZIP.
- The existing local BiRefNet path as an explicit fallback.

The implementation does not add a browser API-key field, automatic provider
fallback, concurrent paid requests, price prediction, Seedream generation,
LLM-based image analysis, or additional cloud vendors.

## Verification

Automated tests use a fake fal gateway and make no paid requests. They verify:

- Cloud mode refuses to start when `FAL_KEY` is missing.
- The fixed endpoint parameters request Light 2K, 2048 resolution, mask-only,
  synchronous mode, and PNG output.
- The API key never appears in an exception, manifest, or UI return value.
- A valid cloud mask produces an output with the exact source dimensions,
  subject scale, position, and opaque RGB pixels.
- A mask-size mismatch is aligned once and records source/returned dimensions.
- Authentication failure stops the batch without retrying every item.
- Pre-acceptance rate limiting uses bounded retries.
- Post-acceptance timeout or network failure never resubmits the job.
- Malformed and empty masks fail or enter review as specified.
- Local processing remains usable without `FAL_KEY`.
- Local painted-mask correction performs no cloud call.
- JPG, JPEG, and PNG downloads continue to match their extensions.

After mock tests pass and the user configures `FAL_KEY`, one explicitly
authorized live call uses the supplied safe image. Acceptance requires the
1024 × 768 input to produce a 1024 × 768 output, preserve the safe's source
position and scale, retain original lock and text pixels, improve edge quality
relative to the current local result, and complete without an unintended
second billable submission.
