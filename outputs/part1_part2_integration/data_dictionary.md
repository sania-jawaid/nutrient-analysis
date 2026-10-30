# Integration data dictionary

> **Generated integration schema snapshot — context added 2026-10-30.** The
> completed [EOF scan](full_scan_summary.json) records **4,881,991 rows /
> 4,881,935 unique barcodes**; integration links **3,075 matched products/images**
> while preserving the **3,083-image** annotation inventory (eight unmatched).
> This is not a join to the separate 50K EDA sample. Conditional unavailable-source
> language below describes schema behavior, not a claim that the current scan is
> missing. See [provenance](provenance.json), the [integration notebook](../../part1_part2_dataset_integration.ipynb)
> and [current workflow](../../README.md#current-workflow--documentation-verified-2026-10-30).
> No Parquet, gold or provenance files were changed; regeneration may overwrite this banner.

Products: unique valid Part2 barcode; nullable Part1 product_name/brand/category/country/serving_size/nutrition_image_url; structured_nutrients is JSON of original source nutrient field names and raw values. Barcode is always a string. Source duplicates are counted and conflicting payloads flagged; first source row is selected deterministically.

Images: image_key=split:source_row is primary key in the recorded dataset revision. Original image_id, barcode_raw, barcode status, URLs, splits, dimensions and annotation status are preserved. Match status does not establish packaging identity. Invalid barcode images are retained without invented product metadata.

Annotations: image_key + token_index is primary key; barcode/image_id/split support audit. Original OCR token, integer bounding_box, gold_bio_id and gold_bio_label remain unchanged. No product metadata is repeated per token. Annotation arrays with missing counterparts remain null and inconsistent image rows are flagged.

Summary: metric/value/scope. Unknown values stay blank while full source is unavailable. Products cover Part2 barcodes only, while full Part1 counts come from a complete disk-backed EOF scan.

Validation: unique products/image keys, no Part2 image loss, annotation foreign keys and counts, annotation Parquet byte hashes, source file stability and original notebook/dashboard hashes.

Warning: barcode, URL or filename ID agreement is not verified pixel/packaging identity; current nutrient metadata is not image-grounded gold. Official train/test splits and cross-split barcode risks are retained.
