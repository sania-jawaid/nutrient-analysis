# Part 1 EDA → Part 2 Extraction: measured observational findings

> **Generated historical analysis — context added 2026-10-30.** All measurements
> below are preserved for the declared **20 aligned official-test images**;
> this is not full-cohort scoring. The newer [checkpoint](../full_cohort_bio/summary.json)
> records 441 completed / 3,075, 1 failed, 2,633 pending and
> `paused_model_load_error`, not a live worker. See the [current workflow](../../README.md#current-workflow--documentation-verified-2026-10-30).
> The [impact notebook](../../part1_eda_impact_on_part2.ipynb) and active
> [analysis module](../../scripts/eda_extraction_impact.py) retain their separate
> cohort/provenance; regeneration may replace this banner. No analysis was rerun here.

No model inference or dataset download was run. Existing gold annotations, prior notebooks and cached predictions are unchanged.

Full background: 4,881,991 product rows. Matched cohort: 3,075 unique products. Evaluated: 20 aligned cached official-test images.

Exact BIO span precision/recall/F1: 0.9263/0.9526/0.9393. Numerical parsing success: 179/217 (82.49%).

## Finding 1

Annotation selection: 3,075 matched products out of 4,881,935 unique full-snapshot barcodes. Serving-size completeness: full 32.5% (1,587,510/4,881,991), matched 23.9% (736/3,075).

**Part 2 implication:** Observed metadata-distribution difference demonstrates that cohort composition differs; it does not prove a causal extraction effect.

**Action:** Keep broad EDA and annotation-selected analysis separate; report serving-basis and selection denominators.

**Evidence type/support:** observed descriptive association — 4,881,991 full rows; 3,075 matched products

## Finding 2

1,313/26,115 image–nutrient pairs have a gold BIO span but no finite structured product value in the mapped field.

**Part 2 implication:** Missing product metadata can coexist with annotated nutrition text; metadata absence must not be counted as model failure or used to drop gold examples.

**Action:** Preserve gold annotations independently; label missing metadata, absent annotations and unrun predictions separately.

**Evidence type/support:** observed paired-source availability — 26,115 gold-positive image–nutrient pairs

## Finding 3

38/217 predicted BIO spans in the aligned cached cohort remain uncertain after conservative numerical parsing.

**Part 2 implication:** High span tagging accuracy does not ensure complete usable nutrition JSON; units, duplicates and multi-number spans need separate review.

**Action:** Report parsing support beside BIO PRF and keep uncertain raw spans instead of guessing corrections.

**Evidence type/support:** observed cached-output diagnostic — 20 evaluated images; 217 predicted spans

## Finding 4

20/127 nutrient-identity/basis/unit-compatible extracted fields disagree with current product values beyond the predeclared 1% numeric tolerance. Product metadata also has 19 flagged quantities across 13 matched products under declared physical/hierarchy review rules.

**Part 2 implication:** Database records and extracted image values are not interchangeable gold; packaging changes, OCR and annotation/model mistakes are competing hypotheses.

**Action:** Prioritize discrepancy examples for image/version adjudication, not automatic model correction.

**Evidence type/support:** observed cross-source discrepancy; causes unverified — 127 comparable fields

## Finding 5

Observed OCR-length strata: 61–120 words: F1 0.8976, n=9 images; ≤60 words: F1 0.9612, n=8 images. Only 20/3075 matched images are evaluated; overall F1 0.9393 on 211 gold spans. Sparse groups are suppressed.

**Part 2 implication:** Image text/layout characteristics coincide with different scores in this small cohort, but nutrient mix and individual difficult examples can confound the association. Category/country strata are often too sparse; no causal driver is established.

**Action:** Maintain the frozen official-test cohort, show every group denominator, and validate hypotheses on additional independently evaluated images before changing preprocessing.

**Evidence type/support:** observed sample limitation; driver claims exploratory — 20 official-test images; zero new inference runs

## Limits and interpretation

- Full counts and fixed-bin distributions are exact row-weighted quantities. Full P1/P25/P50/P75/P99 are approximate from a seed42 uniform reservoir capped at 20,000 finite values per nutrient; matched and 50K quantiles are exact.
- Missing, null, undefined and other explicit placeholder metadata are excluded from completeness/label counts. Multilingual labels are not forced into one inferred country/category.
- Model scores concern supplied corpus OCR, not unseen-image EasyOCR accuracy. Twenty selected test images cannot establish a causal or general population error driver.
- Groups below five entity-contributing images or five gold spans have no reported PRF. Country/category labels overlap; groups are not independent.
- Cross-source discrepancies use identity/unit/basis field conventions and a fixed 1% numeric tolerance. Packaging version, density and image identity are unverified. Product records are never treated as image gold.
- Per-100g physical review bounds are not applied to serving quantities. Statistical tails and hierarchy flags trigger review, not data correction or test-set tuning.
- Standalone smoke/upload prediction caches are excluded; only the declared seeded official-test bundle passes strict image-key/barcode/token/box/gold/coverage checks.
