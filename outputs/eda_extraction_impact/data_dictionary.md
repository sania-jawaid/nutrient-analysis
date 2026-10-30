# EDA → Extraction cached output dictionary

> **Generated artifact/schema snapshot — context added 2026-10-30.** This
> dictionary describes the saved impact analysis, whose prediction source is
> the seeded **20 official-test images**, not the partial full-cohort runner.
> Preserve its field definitions and numerical evidence. See [findings](analysis_findings.md),
> the active [analysis module](../../scripts/eda_extraction_impact.py) and
> [current workflow](../../README.md#current-workflow--documentation-verified-2026-10-30).
> Regeneration can overwrite generated Markdown; no schema/data was changed here.

- representativeness: one source cohort/nutrient; finite coverage counts, min/max, P1/P25/P50/P75/P99 and quantile sample size/approximation flag. source_field_available distinguishes missing columns from zero available product values.
- distribution_bins: exact same-unit fixed-bin counts/fractions per source cohort/nutrient; includes underflow/overflow bins.
- label_distributions and metadata_coverage: source cohort label/field counts and row/product denominator; labels may overlap.
- image_rows: one matched original image key with untouched split, product context, source dimensions/token count and evaluated flag.
- coverage_rows: matched image/nutrient availability states, mapping flag, evaluated flag and separate metadata/gold/model/parsing counts. Unevaluated predictions are null, not zero failure; serving_size has no structured nutrient mapping.
- performance_rows: evaluated image/nutrient exact gold/predicted/true-positive span counts; predicted parseable/uncertain counts/reasons; error token positions retained for union-based deduplicated token totals. Source split is always official test.
- performance_<dimension>: micro aggregation with distinct image count, entity-contributing image support, span counts and sparse-score suppression flag. Multi-label groups overlap.
- anomaly_rows and anomaly_thresholds: raw mapped product values/units/bases, review flags, matched-cohort IQR/P1/P99 thresholds. No values corrected; no thresholds tune the model.
- agreement_rows: parsed predicted span versus current source value; identity/basis/unit/qualifier comparability reasons, numeric difference, predeclared 1% tolerance, reference_is_gold=False. No pooled mixed-unit MAE.
- error_examples: actual original-token gold/prediction BIO disagreements; OCR transcription not independently verified.
- key_findings.csv and analysis_findings.md: five computed observations with interpretation, actions, evidence and support.
- manifest.json/prediction_audit.json/full_background_manifest.json: model-cache cohort provenance, strict alignment checks, protected-input hashes, full-source fingerprint, version, quantile method, no-inference guarantee and limitations.
