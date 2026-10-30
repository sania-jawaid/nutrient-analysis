# Manual French Nutrition Panel — OCR Validation Summary

> **Generated one-image OCR/rule snapshot — context added 2026-10-30.** Preserve
> all detections, CER/WER and field tables below under their original scoring
> conventions. This is not the later EasyOCR→ONNX manual-reference run and is
> not an all-image OCR benchmark. The [later follow-up](../part2/bio_remaining_tasks_findings.md)
> uses its own run and field matching; figures must not be pooled or silently
> replaced. See the [OCR notebook](../../part2/part2_ocr_validation.ipynb) and
> [current workflow](../../README.md#current-workflow--documentation-verified-2026-10-30).
> No OCR, model, notebook, gold or measured output was rerun/changed here;
> regeneration can overwrite this generated Markdown banner.

- Product: `241013128`
- Image: `data/manual_samples/241013128_nutrition.jpg` (400×253)
- OCR status: `success`
- OCR detections: 11
- Manual gold fields: 16
- Exact nutrient-value-unit-basis matches: 1
- CER/WER: 0.09908256880733946 / 0.4423076923076923

This is a one-image manual-validation fixture, not aggregate model performance. OCR ran with CPU EasyOCR using French and English. Its detector recognized long text lines and made visible character errors; the rule baseline therefore leaves many fields unresolved. No model-performance claim is made from one image.

## Field comparison

```text
     nutrient  gold_basis predicted_basis  basis_match  gold_value gold_unit_canonical gold_unit_original  predicted_value predicted_unit_canonical predicted_unit_original  absolute_difference_canonical_units  value_within_tolerance  exact_original_unit_match  exact_nutrient_value_unit_basis_match             status
       sugars    per_100g     unspecified        False   21.700000                   g                  g        21.700000                        g                       g                                  0.0                    True                       True                                  False     basis_mismatch
       energy per_serving     per_serving         True  213.432122                kcal                 kJ       213.432122                     kcal                      kj                                  0.0                    True                       True                                   True        exact_match
       energy    per_100g            None        False  237.093690                kcal                 kJ              NaN                     None                    None                                  NaN                   False                      False                                  False missing_prediction
       energy    per_100g            None        False  237.000000                kcal               kcal              NaN                     None                    None                                  NaN                   False                      False                                  False missing_prediction
          fat    per_100g            None        False    9.500000                   g                  g              NaN                     None                    None                                  NaN                   False                      False                                  False missing_prediction
saturated_fat    per_100g            None        False    4.200000                   g                  g              NaN                     None                    None                                  NaN                   False                      False                                  False missing_prediction
carbohydrates    per_100g            None        False   32.000000                   g                  g              NaN                     None                    None                                  NaN                   False                      False                                  False missing_prediction
     proteins    per_100g            None        False    4.600000                   g                  g              NaN                     None                    None                                  NaN                   False                      False                                  False missing_prediction
         salt    per_100g            None        False    0.360000                   g                  g              NaN                     None                    None                                  NaN                   False                      False                                  False missing_prediction
       energy per_serving            None        False  213.000000                kcal               kcal              NaN                     None                    None                                  NaN                   False                      False                                  False missing_prediction
          fat per_serving            None        False    8.600000                   g                  g              NaN                     None                    None                                  NaN                   False                      False                                  False missing_prediction
saturated_fat per_serving            None        False    3.800000                   g                  g              NaN                     None                    None                                  NaN                   False                      False                                  False missing_prediction
carbohydrates per_serving            None        False   28.800000                   g                  g              NaN                     None                    None                                  NaN                   False                      False                                  False missing_prediction
       sugars per_serving            None        False   19.500000                   g                  g              NaN                     None                    None                                  NaN                   False                      False                                  False missing_prediction
     proteins per_serving            None        False    4.100000                   g                  g              NaN                     None                    None                                  NaN                   False                      False                                  False missing_prediction
         salt per_serving            None        False    0.320000                   g                  g              NaN                     None                    None                                  NaN                   False                      False                                  False missing_prediction
```

## Failure review

```text
product_id                                  image_path                        failure_type                                basis      nutrient                             gold           prediction         root_cause severity                                                              action
 241013128 data/manual_samples/241013128_nutrition.jpg serving_vs_100g_or_basis_assignment gold=per_100g; predicted=unspecified        sugars                21.7 g @ per_100g 21.7 g @ unspecified     basis_mismatch   review Inspect OCR text, units, row association, and serving-basis column.
 241013128 data/manual_samples/241013128_nutrition.jpg            missed_nutrient_or_value        gold=per_100g; predicted=None        energy 237.09369024856596 kJ @ per_100g     None None @ None missing_prediction   review Inspect OCR text, units, row association, and serving-basis column.
 241013128 data/manual_samples/241013128_nutrition.jpg            missed_nutrient_or_value        gold=per_100g; predicted=None        energy            237.0 kcal @ per_100g     None None @ None missing_prediction   review Inspect OCR text, units, row association, and serving-basis column.
 241013128 data/manual_samples/241013128_nutrition.jpg            missed_nutrient_or_value        gold=per_100g; predicted=None           fat                 9.5 g @ per_100g     None None @ None missing_prediction   review Inspect OCR text, units, row association, and serving-basis column.
 241013128 data/manual_samples/241013128_nutrition.jpg            missed_nutrient_or_value        gold=per_100g; predicted=None saturated_fat                 4.2 g @ per_100g     None None @ None missing_prediction   review Inspect OCR text, units, row association, and serving-basis column.
 241013128 data/manual_samples/241013128_nutrition.jpg            missed_nutrient_or_value        gold=per_100g; predicted=None carbohydrates                32.0 g @ per_100g     None None @ None missing_prediction   review Inspect OCR text, units, row association, and serving-basis column.
 241013128 data/manual_samples/241013128_nutrition.jpg            missed_nutrient_or_value        gold=per_100g; predicted=None      proteins                 4.6 g @ per_100g     None None @ None missing_prediction   review Inspect OCR text, units, row association, and serving-basis column.
 241013128 data/manual_samples/241013128_nutrition.jpg            missed_nutrient_or_value        gold=per_100g; predicted=None          salt                0.36 g @ per_100g     None None @ None missing_prediction   review Inspect OCR text, units, row association, and serving-basis column.
 241013128 data/manual_samples/241013128_nutrition.jpg            missed_nutrient_or_value     gold=per_serving; predicted=None        energy         213.0 kcal @ per_serving     None None @ None missing_prediction   review Inspect OCR text, units, row association, and serving-basis column.
 241013128 data/manual_samples/241013128_nutrition.jpg            missed_nutrient_or_value     gold=per_serving; predicted=None           fat              8.6 g @ per_serving     None None @ None missing_prediction   review Inspect OCR text, units, row association, and serving-basis column.
 241013128 data/manual_samples/241013128_nutrition.jpg            missed_nutrient_or_value     gold=per_serving; predicted=None saturated_fat              3.8 g @ per_serving     None None @ None missing_prediction   review Inspect OCR text, units, row association, and serving-basis column.
 241013128 data/manual_samples/241013128_nutrition.jpg            missed_nutrient_or_value     gold=per_serving; predicted=None carbohydrates             28.8 g @ per_serving     None None @ None missing_prediction   review Inspect OCR text, units, row association, and serving-basis column.
 241013128 data/manual_samples/241013128_nutrition.jpg            missed_nutrient_or_value     gold=per_serving; predicted=None        sugars             19.5 g @ per_serving     None None @ None missing_prediction   review Inspect OCR text, units, row association, and serving-basis column.
 241013128 data/manual_samples/241013128_nutrition.jpg            missed_nutrient_or_value     gold=per_serving; predicted=None      proteins              4.1 g @ per_serving     None None @ None missing_prediction   review Inspect OCR text, units, row association, and serving-basis column.
 241013128 data/manual_samples/241013128_nutrition.jpg            missed_nutrient_or_value     gold=per_serving; predicted=None          salt             0.32 g @ per_serving     None None @ None missing_prediction   review Inspect OCR text, units, row association, and serving-basis column.
```

## Provenance and interpretation

The reference transcription and 16 structured nutrient entries were manually reviewed earlier and loaded from the project evaluation files. EasyOCR output is kept separate. OCR confidence is a recognizer score, not calibrated field-association confidence. No OCR-to-BIO token-index comparison is made.