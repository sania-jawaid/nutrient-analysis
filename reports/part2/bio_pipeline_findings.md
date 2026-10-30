# Part 2: measured local BIO pipeline improvements

> **Historical measured seeded-20 benchmark — context added 2026-10-30.** All
> scores, coverage, timing, memory and error evidence below are retained from
> this run, not newly executed measurements. The shared coarse comparison is
> now explained as **Basic Nutrient Detection** (rules/model on identical tokens
> and projected gold); fine-grained model-only scoring is **Detailed Nutrient
> Identification**. The separate [full-cohort checkpoint](../../outputs/full_cohort_bio/summary.json)
> records **441/3,075 completed, 1 failed, 2,633 pending**, status
> `paused_model_load_error`, not an active worker or replacement benchmark.
> See the [current workflow](../../README.md#current-workflow--documentation-verified-2026-10-30)
> and [cache-only Dashboard2](../../app_clean_README.md). Original-app references
> below concern port 8501; the clean app on 8502 never runs inference.

## Scope and reproducibility

Existing Part 1 pages, notebooks, source data and rule extraction remain in place. The local Open Food Facts LayoutLMv3 ONNX checkpoint runs on CPU; no training, checkpoint download, paid inference, or model quantization was performed. EasyOCR's default dynamic quantization is explicitly disabled.

- Official test split: 199 rows; sampled 20 with Python `random.Random(42).sample`, sorted for sequential processing.
- Selected indices: 6, 7, 8, 22, 23, 26, 28, 35, 55, 57, 59, 62, 70, 108, 139, 151, 163, 173, 188, 189.
- Completed: **20/20**, with **zero per-image failures**. An initial metadata request returned HTTP 502; one retry succeeded.
- Windows: 22 total; overlapping first-subword logits are averaged at original word positions. Adaptive smaller windows handle tokenizer truncation. An image is rejected if coverage cannot be completed.
- Token coverage: **1,551/1,551 (100%)**. Gold entity coverage: **211/211 (100%)**.
- Separate real regression on the original three images: **453/453** words covered, previously 391/453. Individual counts: 87/87, 242/242 and 124/124.

## Actual model measurements

| Task | Precision | Recall | F1 | Token accuracy |
|---|---:|---:|---:|---:|
| Fine-grained exact BIO spans | 0.9263 | 0.9526 | 0.9393 | 0.9858 |
| Model, shared coarse VALUE/BASIS labels | 0.9767 | 0.9953 | 0.9859 | 0.9955 |
| Existing rule baseline, same coarse labels | 0.4680 | 0.7962 | 0.5895 | 0.8375 |

The fine-grained model matched 201 exact spans, predicting 217 against 211 gold spans. The coarse comparison uses identical OCR tokens, boxes and projected gold labels for both methods; raw rule labels are not compared to the incompatible 71-label ontology.

Mean processor + ONNX inference time: **3.26 seconds/image**. Mean sampled process RSS after inference: **2,035 MB**; maximum sampled post-image RSS: **2,078 MB**, not an instrumented peak. Mean per-image RSS change: +3.34 MB. Memory includes the resident session, and available system RAM changes are affected by other applications. Initial loading requires 2.5 GiB available; resident-session evaluation stops below 0.75 GiB.

## Structured fields versus official annotations

Exact normalized nutrient/basis/unit/value/qualifier tuples were decoded independently from predicted labels and official `ner_tags` on the same tokens:

- Matched: **174**; predicted parseable fields: **179**; gold parseable fields: **187**.
- Field precision **0.9721**, recall **0.9305**, F1 **0.9508**.
- **24 uncertain gold spans excluded**, explicitly disclosed; 38 predicted spans flagged for review.

This is annotation-derived evaluation, not independent visual transcription or OCR accuracy. Qualifiers such as `<` are retained; duplicate semantic fields and malformed spans are not silently resolved. Raw tags and text remain available in JSON.

## Actual errors and limitations

There were **22 token disagreements**: 15 wrong nutrient/basis labels, six false-positive tokens and one BIO-boundary mismatch.

- Barcode 3760027470152: `14,92 g` was annotated as proteins but predicted as fat; several subsequent nutrient values were shifted among row labels. A `120 g` quantity outside annotated nutrition fields was predicted as salt. This is label/layout confusion; it is not proof of an OCR transcription error.
- Barcode 7622210645128: `56g/` was annotated as sugars but predicted as carbohydrates; `0,9` was assigned sugars instead of fiber.
- Barcode 4099100042832: `1g` began a new span when the annotation continued an added-sugars serving span.
- Barcode 7613036278621: `255 kcal` was predicted as serving energy where corpus gold was O. Without adjudication, this is a corpus disagreement rather than proof that either source is visually correct.
- Parser diagnostics include `Og`, missing units, multi-number spans, damaged energy-unit text and conflicting repeated fields. No automatic OCR correction was invented.

## End-to-end EasyOCR check

The upload path was actually exercised on a cached image without supplied OCR or gold labels: **82 OCR words, 82 covered model positions, four parsed fields and eight uncertain spans**. CPU OCR took approximately 2.35 seconds and ONNX inference 3.23 seconds. Words from multiword detections share the actual detection box; fabricated word-level geometry is not used.

Two harder image checks returned no nutrition entities after poor OCR. Rotation, blur, language and grouping remain material failure modes. The application warns when no unambiguous fields are returned. This integration check does not establish accuracy on unseen uploads, and no positional BIO or field accuracy is claimed for EasyOCR tokens.

## Separate Open Food Facts API reference check

Actual metadata for the first 10 evaluated barcodes was fetched and cached. All 10 products were found. Product-level exact agreement varies, and the per-product comparisons are saved separately in [the machine-readable measured report](bio_measured_evaluation.json). API values can change, differ from photographed packaging, omit serving values, or contain extra nutrition fields; they are **not image-verified gold** and are never pooled into BIO metrics. API-missing references are unscored. Missing model fields relative to available API metadata are labeled explicitly as metadata-only omissions.

## Artifacts

- [Executed notebook](../../part2/part2_layoutlm_bio_tagging.ipynb): inference, coverage regression, real errors, OCR smoke check and API comparisons.
- [Machine-readable measurements](bio_measured_evaluation.json): scores, sampled indices, errors and separate reference comparisons.
- The dashboard loads its versioned local prediction bundle without opening weights and remains available on port 8501.