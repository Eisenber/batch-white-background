# OpenAI GPT Image 2 High-Quality White-Background Design

## Goal

Replace the fal.ai cloud mask mode with an optional OpenAI GPT Image 2
high-quality generation mode. Keep the existing local BiRefNet mode as the
default offline path. GPT mode creates a complete white-background product
image, while local mode keeps the existing mask-and-composite pipeline.

The final downloadable image must always have exactly the same pixel width,
pixel height, and aspect ratio as its source image. The product should remain
at its original position, scale, orientation, and apparent proportions.

## Product Modes

The workbench exposes exactly two processing engines:

- `本地离线 · BiRefNet` (`local`), selected by default.
- `高质量生成 · GPT Image 2` (`openai`), selected explicitly by the user.

The fal.ai option, implementation, tests, dependency, and active design
documentation are removed. Git history is not rewritten.

Local mode remains behaviorally unchanged and supports batches of up to 50
images. GPT mode supports batches of up to 10 images and processes them
sequentially.

## Architecture

`BatchOptions.processing_engine` accepts `local` or `openai`. The batch layer
selects one of two deliberately separate processing paths:

1. Local mode obtains a segmentation mask from the existing local session,
   applies the existing conservative geometry and glare corrections, and
   composites aligned source pixels over white.
2. GPT mode delegates one source image to an `OpenAIImageSession`, which returns
   a complete edited RGB image. It does not pass the generated image through
   the local mask pipeline and does not apply automatic geometry or glare
   correction afterward.

The OpenAI adapter has one responsibility: validate and prepare one source,
submit one image-edit request, validate the response, and return a decoded
image. It does not write output files or manage a batch. The batch layer keeps
ownership of filenames, manifests, progress, failure isolation, and direct
JPG/JPEG/PNG downloads.

## OpenAI Request

GPT mode uses the OpenAI Image Edit API with model `gpt-image-2`. Each image is
submitted once with high quality and an opaque white-background instruction.
The prompt requires the model to:

- replace only the surrounding environment and supporting surface with pure
  white `#FFFFFF`;
- preserve the safe's original position, scale, proportions, orientation, and
  framing;
- preserve all product text, number buttons, fingerprint reader, lock hardware,
  seams, colors, reflections, surface texture, and construction details;
- add or remove no product components;
- retain only a subtle natural contact shadow when necessary to keep the
  product grounded;
- produce a professional e-commerce product image without decorative elements.

GPT Image 2 is a generative editor, so these instructions reduce but cannot
eliminate the risk of altered product details. Every GPT result is labeled
`GPT 成品 · 需人工复核`.

## Input Preparation and Output Geometry

The source is EXIF-transposed before dimensions are recorded. It is converted
to a supported RGB upload without changing its visible orientation. Inputs that
cannot be represented within the API file-size requirements are rejected
locally before a paid request is submitted.

The adapter chooses a legal GPT Image 2 output size whose aspect ratio is
closest to the oriented source. It does not crop the returned composition.
After decoding, the returned image is converted to RGB and resized once to the
exact oriented-source dimensions with a high-quality resampler. Therefore the
downloaded width, height, and aspect ratio always match the source exactly,
even when the API cannot generate that exact pixel size.

The manifest records the source dimensions, requested GPT dimensions, returned
dimensions, final dimensions, model, and that a final resize occurred. It never
records credentials.

## Authentication and Secret Handling

The adapter reads `OPENAI_API_KEY` only from the server process environment.
The key is never accepted through the browser, stored in task state, written to
the batch manifest, included in logs, or committed to the repository.

Missing credentials fail before any network request. Authentication or account
permission failure is treated as fatal for the session so later items in the
same batch are not submitted with the same unusable credential.

## Cost and Retry Safety

GPT mode is never the default. Its UI copy states that the original is uploaded
to OpenAI and that each image consumes paid API usage.

GPT batches are capped at 10 and processed sequentially. The client is
configured with automatic retries disabled. Each image-edit request is
submitted at most once. A connection interruption, timeout, or ambiguous
submission result is reported without automatic retry because the server may
already have accepted a billable request.

The application does not automatically fall back from GPT to local processing.
This prevents a lower-quality local result from being mistaken for a GPT
result. The user may explicitly start a new batch after reviewing an error.

## Error Handling

Errors are converted to concise Chinese messages without including request
payloads or credentials. The UI distinguishes at least:

- missing `OPENAI_API_KEY`;
- invalid credentials or insufficient account permission;
- insufficient API balance or quota;
- rate limiting;
- unsupported, oversized, or malformed input;
- content-policy rejection;
- request timeout or ambiguous connection failure;
- invalid or missing returned image data;
- OpenAI service failure.

A failure for one ordinary image is recorded on that item while already
successful items remain downloadable. A fatal authentication/account error
prevents subsequent paid submissions in that batch. No error path silently
retries or switches engines.

## User Interface

The engine selector contains only the local and GPT modes and defaults to local.
Selecting GPT displays a visible notice that the source will be uploaded, the
request is paid, and generated edits can alter text, buttons, lock hardware, or
surface details.

The UI enforces the engine-specific batch limits before processing:

- local: 1–50 images;
- GPT Image 2: 1–10 images.

Progress messages distinguish source preparation, OpenAI upload, GPT editing,
source-size restoration, file encoding, completion, and failure. All fal.ai
labels, instructions, and links are removed.

The output format selector remains JPG, JPEG, or PNG. Results remain individual
direct-download files; CSV reports and ZIP archives are not introduced.

## Dependency and Removal Scope

Implementation removes the fal.ai gateway module and its tests. Any dependency
file used only by fal.ai is removed or replaced with the minimal OpenAI client
dependency required by this mode. No fal.ai key, endpoint, model label, or UI
copy remains in active code.

Unrelated rembg CLI and HTTP behavior is unchanged. Existing local models and
sessions remain available.

## Testing

Automated tests use a fake OpenAI gateway and never make paid requests. They
verify:

- missing `OPENAI_API_KEY` fails before network access;
- credentials never appear in errors, manifests, or UI state;
- the model is exactly `gpt-image-2` with high-quality image editing;
- the preservation and pure-white requirements are present in the prompt;
- each image is submitted at most once and automatic retries are disabled;
- GPT batches above 10 are rejected locally, while local batches up to 50 remain
  accepted;
- authentication failure prevents later submissions in the same batch;
- ordinary single-image failure preserves other successful outputs;
- legal request dimensions are chosen near the source aspect ratio;
- returned images are restored to the exact source dimensions;
- JPG, JPEG, and PNG direct downloads still work;
- GPT results carry the manual-review label;
- fal.ai modules, options, dependencies, and tests are absent;
- all existing local fidelity and direct-download tests still pass.

After automated tests, the local Gradio interface is smoke-tested. A real paid
sample request is outside automated testing and occurs only after the user
configures `OPENAI_API_KEY` and explicitly authorizes one sample call.

## Acceptance Criteria

The design is complete when the application offers only local BiRefNet and GPT
Image 2 modes, defaults to local, removes active fal.ai support, and safely
generates individually downloadable white-background JPG/JPEG/PNG files. GPT
outputs must return to the exact source dimensions, be labeled for manual
review, and never be automatically retried or silently replaced with local
results.
