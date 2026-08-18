# Direct Image Downloads Design

## Goal

Replace the downloadable CSV report and ZIP archive with direct downloads of
the processed image files. A batch uses one user-selected output format: JPG,
JPEG, or PNG.

## User Experience

The product workbench adds an output-format control beside the quality control.
Its choices are JPG, JPEG, and PNG, with JPG selected by default.

After processing, the download section displays a multi-file list containing
every successfully generated image. Users can download individual images or
select multiple files through the Gradio file component. No CSV report or ZIP
archive is created or offered.

Output names retain the existing status suffix:

- A normal result uses `<source-stem>_white.<extension>`.
- A result requiring review uses `<source-stem>_review.<extension>`.
- JPG uses `.jpg`, JPEG uses `.jpeg`, and PNG uses `.png`.

JPG and JPEG use the same JPEG encoding. PNG uses lossless RGB PNG encoding.

## Architecture

### Processing options and serialization

`BatchOptions` gains a validated output-format field. `ProcessingResult`
provides format-aware image serialization instead of a JPEG-only method.

JPEG serialization keeps the existing quality, no-chroma-subsampling, and
optimization settings. PNG serialization writes an optimized RGB PNG. The
JPEG quality setting is ignored for PNG.

### Batch output

`BatchResult` exposes the task directory, batch items, and the paths of all
generated output images. It no longer exposes a ZIP path.

The batch processor continues to use the `processed` and `review` directories
so status remains visible in internal state and correction assets stay
organized. It does not create `report.csv` or `safe_white_images.zip`.

`manifest.json` and the `.edit` directory remain private implementation data.
They are required to align source pixels and masks for local correction and
are not included in the download list.

### User interface

The batch callback accepts the selected output format and returns a list of
generated image paths. A Gradio multi-file component replaces the ZIP download
button.

The processing details table remains unchanged so users can still see status,
elapsed time, applied steps, warnings, and errors. Failed items are shown in
the table but omitted from the download list because they have no output file.

### Local mask correction

Local correction overwrites the selected item's existing output file using the
batch's chosen format. After correction, the callback refreshes both the
generated-image gallery and the complete downloadable-file list. It does not
rebuild a report or archive.

## Data Flow

1. The user uploads 1-50 supported source images.
2. The user selects processing quality and one output format for the batch.
3. The batch processor segments and composes each white-background result.
4. Each successful result is serialized in the selected format and stored in
   its status directory.
5. The UI receives all output paths and exposes them as direct downloads.
6. If the user applies a mask correction, the matching output file is
   overwritten and the download list is refreshed.

## Error Handling

- An unsupported output format is rejected before model inference starts.
- A failed item does not stop the rest of the batch.
- A failed item remains visible in the details table and has no download path.
- A correction without a selected item or painted region keeps the existing
  user-facing error behavior.
- If an image cannot be serialized, that item is marked failed and is excluded
  from downloads.

## Compatibility and Scope

- Input support remains PNG, JPG, and JPEG.
- Output support is JPG, JPEG, and PNG.
- White-canvas dimensions, subject ratio, model selection, quality warnings,
  preview galleries, and local-only inference behavior are unchanged.
- No automatic browser multi-download behavior is added, avoiding popup and
  multiple-download restrictions.
- No CSV or ZIP compatibility mode is retained.

## Verification

- Run Python compilation over the `rembg` package.
- Verify output extensions for JPG, JPEG, and PNG.
- Decode each written file to confirm that its actual encoding matches its
  extension.
- Use a fake segmentation session and synthetic source image to verify batch
  processing without downloading an ONNX model.
- Verify normal and review output naming.
- Verify the direct-download path list excludes failed items and private edit
  assets.
- Verify local correction overwrites the selected output and returns the
  refreshed list without creating a CSV or ZIP.
