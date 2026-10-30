# Pre-inference readiness: measured evidence

> **Generated pre-inference baseline — context added 2026-10-30.** Preserve all
> measured counts below: 3,083 annotation records versus only **25 locally cached
> pixel-measured images**. Do not generalize convenience-sample pixel rates or
> recalculate this baseline from predictions. Later inference progress belongs
> to separate [checkpoint results](../full_cohort_bio/summary.json), currently
> `paused_model_load_error`, not an active worker. See the [readiness notebook](../../part2_preprocessing_eda.ipynb)
> and [current workflow](../../README.md#current-workflow--documentation-verified-2026-10-30).
> No audit, images, annotations or notebook outputs were rerun/changed here;
> generation may overwrite this context banner.

{
  "total_annotated_images": 3083,
  "matched_products": 3075,
  "unmatched_images": 8,
  "token_annotations": 309190,
  "pixel_measured_images": 25,
  "tokenizer_measured_images": 3083,
  "sliding_window_images": 455,
  "proxy_experiment_images": 8
}

## Input longer than 512 LayoutLMv3 subwords including specials
- Affected: 455/3083 (14.76%) tokenizer-measured annotated images
- Class: Required preprocessing
- Action: Retain existing overlapping sliding windows and word_ids alignment; no single-pass truncation for affected images.
- Benefit: Full original-word coverage rather than silent truncation.
- Validation: Assert every original word is covered; validate boundary spans and overlap merging without tuning gold.

## Invalid/alignment-risk token-box-label records
- Affected: 47/3083 (1.52%) all annotated images
- Class: Required preprocessing
- Action: Validate aligned tokens/boxes/labels and [0,1000] nondegenerate integer geometry before model encoding; quarantine failures, never silently clamp/reorder gold.
- Benefit: Prevent invalid model inputs while preserving annotations.
- Validation: Read-only alignment, coordinate and tokenizer word_ids checks. If zero affected, retain checks; no corrective transform justified.

## EXIF orientation transformation indicated
- Affected: 0/25 (0.00%) cached pixel-measured images only
- Class: Required preprocessing
- Action: Retain EXIF-aware image loading for new OCR inputs; if transforming already annotated images, coordinate-transform boxes consistently or preserve original geometry.
- Benefit: Avoid metadata-directed orientation mismatch.
- Validation: Compare original/oriented dimensions and transformed box overlays; do not infer rotation from aspect ratio.

## Short image side below 400 pixels
- Affected: 327/3083 (10.61%) all images with source dimensions
- Class: Data-quality review flag
- Action: Inspect legibility; do not assume upscaling restores text. Prefer original higher-resolution source only if already available.
- Benefit: Identify likely text-detail limitations before OCR.
- Validation: Inspect representative panels and validate OCR against manual text, not sharpness proxies alone.

## Source maximum side exceeds existing 2000px OCR cap
- Affected: 541/3083 (17.55%) all images with source dimensions
- Class: Optional preprocessing
- Action: Retain aspect-preserving maximum-side resize as a resource-control option, not an accuracy claim. Do not upscale small sources; preserve the model processor saved image settings.
- Benefit: Bound CPU OCR workload while avoiding unnecessary upscaling or distorted aspect ratios.
- Validation: Compare before/after legibility and manual-reference OCR quality; verify scaled geometry/normalized boxes. Existing corpus model inputs must retain image/box alignment.

## Low grayscale contrast proxy
- Affected: 3/25 (12.00%) cached pixel-measured convenience sample only
- Class: Optional preprocessing
- Action: Consider contrast normalization only on reviewed low-contrast inputs; default 1.15 contrast is not justified for every image.
- Benefit: Potential readability benefit is a hypothesis; current experiment measures proxies only.
- Validation: Paired before/after OCR on manually transcribed held-out images; report CER/WER with identical OCR settings before adoption.

## Low sharpness/Laplacian proxy
- Affected: 0/25 (0.00%) cached pixel-measured convenience sample only
- Class: Optional preprocessing
- Action: Review small/blurred panels; do not apply unconditional sharpening or blur correction.
- Benefit: Potential readability benefit is a hypothesis; current experiment measures proxies only.
- Validation: Paired before/after OCR on manually transcribed held-out images; report CER/WER with identical OCR settings before adoption.

## Dark luminance proxy
- Affected: 1/25 (4.00%) cached pixel-measured convenience sample only
- Class: Optional preprocessing
- Action: Consider brightness correction only if visual review confirms underexposure.
- Benefit: Potential readability benefit is a hypothesis; current experiment measures proxies only.
- Validation: Paired before/after OCR on manually transcribed held-out images; report CER/WER with identical OCR settings before adoption.

## Bright luminance proxy
- Affected: 0/25 (0.00%) cached pixel-measured convenience sample only
- Class: Optional preprocessing
- Action: Review white-background versus overexposure; avoid automatic darkening from mean brightness alone.
- Benefit: Potential readability benefit is a hypothesis; current experiment measures proxies only.
- Validation: Paired before/after OCR on manually transcribed held-out images; report CER/WER with identical OCR settings before adoption.

## Small OCR envelope within image (<40% of normalized area)
- Affected: 96/3083 (3.11%) all annotated images with valid boxes
- Class: Data-quality review flag
- Action: Inspect candidate margins; do not auto-crop from all-text OCR envelope. Use verified nutrition-table localization if a crop is later justified, remapping boxes.
- Benefit: Potentially reduce irrelevant packaging text without deleting nutrient context.
- Validation: Manually verify crop retains every target span; coordinate transform/inverse tests; OCR quality comparison.

## Decimal-comma numeric text
- Affected: 2463/3083 (79.89%) all annotated images (at least one occurrence)
- Class: Required preprocessing
- Action: Retain decimal-comma normalization in downstream numeric parser; never replace the original OCR/gold text.
- Benefit: Avoid unsupported OCR/annotation changes and retain traceable input alignment.
- Validation: Inspect token/box/image examples; record reversible proposals separately and keep official evaluation untouched.

## Multiple numeric mentions in one OCR token
- Affected: 890/3083 (28.87%) all annotated images (at least one occurrence)
- Class: Data-quality review flag
- Action: Preserve OCR token geometry and flag segmentation for review; do not split annotated gold words without an explicit reversible mapping.
- Benefit: Avoid unsupported OCR/annotation changes and retain traceable input alignment.
- Validation: Inspect token/box/image examples; record reversible proposals separately and keep official evaluation untouched.

## Number and unit merged in one OCR token
- Affected: 2565/3083 (83.20%) all annotated images (at least one occurrence)
- Class: Required preprocessing
- Action: Use unit-aware parsing that accepts merged values/units; retain original word_ids and raw OCR.
- Benefit: Avoid unsupported OCR/annotation changes and retain traceable input alignment.
- Validation: Inspect token/box/image examples; record reversible proposals separately and keep official evaluation untouched.

## Malformed numeric glyph/separator review
- Affected: 386/3083 (12.52%) all annotated images (at least one occurrence)
- Class: Data-quality review flag
- Action: Flag raw strings; do not replace O→0 or g→9 automatically. Validate any correction using image evidence.
- Benefit: Avoid unsupported OCR/annotation changes and retain traceable input alignment.
- Validation: Inspect token/box/image examples; record reversible proposals separately and keep official evaluation untouched.

## Backward vertical reading-order jumps
- Affected: 2825/3083 (91.63%) all annotated images (at least one occurrence)
- Class: Data-quality review flag
- Action: Inspect multi-column reading order before any sorting. Preserve original corpus order for aligned annotation evaluation.
- Benefit: Avoid unsupported OCR/annotation changes and retain traceable input alignment.
- Validation: Inspect token/box/image examples; record reversible proposals separately and keep official evaluation untouched.

## Duplicate bounding boxes
- Affected: 2/3083 (0.06%) all annotated images (at least one occurrence)
- Class: Data-quality review flag
- Action: Inspect shared/duplicated detection geometry; never discard gold tokens solely because boxes repeat.
- Benefit: Avoid unsupported OCR/annotation changes and retain traceable input alignment.
- Validation: Inspect token/box/image examples; record reversible proposals separately and keep official evaluation untouched.

## High-IoU overlapping box pairs
- Affected: 47/3083 (1.52%) all annotated images (at least one occurrence)
- Class: Data-quality review flag
- Action: Review segmentation/shared-box context; no automatic NMS or box merge on gold annotations.
- Benefit: Avoid unsupported OCR/annotation changes and retain traceable input alignment.
- Validation: Inspect token/box/image examples; record reversible proposals separately and keep official evaluation untouched.

## Invalid BIO continuation transitions
- Affected: 0/3083 (0.00%) all annotated images (at least one occurrence)
- Class: Data-quality review flag
- Action: Quarantine/adjudicate labels for review; retain original tags, never auto-repair evaluation gold.
- Benefit: Avoid unsupported OCR/annotation changes and retain traceable input alignment.
- Validation: Inspect token/box/image examples; record reversible proposals separately and keep official evaluation untouched.

## Missing serving-size context
- Affected: 2339/3075 (76.07%) unique barcode-matched products
- Class: Data-quality review flag
- Action: Keep per-serving and per-100g values separate; never infer a serving mass or convert without explicit evidence.
- Benefit: Avoid ungrounded measurement-basis conversion before/after extraction.
- Validation: Require explicit parsed serving mass/volume and verify packaging basis before normalization.

## Implausible/negative structured product quantities
- Affected: 11/3075 (0.36%) unique barcode-matched products
- Class: Post-inference validation
- Action: Use unit/basis-aware validation as a downstream review guard, not a reason to alter image pixels or gold labels.
- Benefit: Avoid treating unverified database anomalies as preprocessing targets or image truth.
- Validation: Check units, basis and packaging version manually; no automatic replacement of extracted values.

## Limits
- Pixel indicators use cached availability sample only, not all images.
- OCR diagnostics concern supplied corpus OCR, not unseen-image EasyOCR error rates.
- Aspect ratio is not text rotation; EXIF orientation alone cannot detect sideways/upside-down text.
- OCR envelope is not a nutrition-table annotation; crop flags cannot justify automatic crops.
- Sharpness/brightness/contrast thresholds are explicit uncalibrated review screens, not learned quality labels.
- Tokenizer lengths are exact local-tokenizer text lengths with untouched boxes, not geometry approval or model forward passes; malformed box structures/alignment remain unmeasured.
- Per-100g database field units/basis are conventions, not packaging-verified truth.
- No prediction, F1, performance or post-inference output influences readiness counts.
- Contrast/resize test measures pixel proxies only; no OCR benefit has been demonstrated.