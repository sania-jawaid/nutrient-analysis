"""Check annotated product barcodes against the Open Food Facts product index.

Uses the existing Hugging Face row retrieval and barcode-normalization helpers,
then performs exact-code batch searches on Search-a-licious. No images or
product-level linked dataset are downloaded or written.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_dataset_overlap import (
    DEFAULT_API_ROOT,
    DEFAULT_DATASET,
    fetch_split_rows,
    get_dataset_metadata,
    get_split_counts,
    normalize_id,
)

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ENDPOINT = "https://search.openfoodfacts.org/search"
REPORT = ROOT / "reports" / "complete_off_barcode_overlap.md"
BATCH_SIZE = 50
REQUEST_DELAY_SECONDS = 0.5
USER_AGENT = "NutritionOverlapAudit/1.0 (research; contact: project-local)"


def get_exact_codes(codes: list[str]) -> tuple[set[str], str | None]:
    """Query a batch of exact barcode terms and return hits or an error."""
    query = "code:(" + " OR ".join(codes) + ")"
    params = urlencode(
        {
            "q": query,
            "fields": "code",
            "page_size": len(codes),
            "page": 1,
        }
    )
    request = Request(
        f"{SEARCH_ENDPOINT}?{params}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    for attempt in range(3):
        try:
            with urlopen(request, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("timed_out"):
                return set(), "search service reported timed_out=true"
            if not payload.get("is_count_exact", False):
                return set(), "search service did not certify an exact result count"
            if payload.get("errors"):
                return set(), f"search service returned errors: {payload['errors']}"
            hits = payload.get("hits")
            if not isinstance(hits, list):
                return set(), "search response omitted the hits array"
            returned = {str(hit.get("code", "")).strip() for hit in hits if hit.get("code") is not None}
            unexpected = returned - set(codes)
            if unexpected:
                return set(), f"exact query returned unexpected codes: {sorted(unexpected)[:3]}"
            # Search count must agree with the returned page; otherwise this
            # batch is inconclusive rather than treated as a negative lookup.
            if int(payload.get("count", -1)) != len(returned):
                return set(), "search result count and returned exact-code hits differ"
            return returned, None
        except HTTPError as error:
            if error.code == 503 and attempt < 2:
                time.sleep(2**attempt + 0.5)
                continue
            return set(), f"HTTP {error.code}"
        except (URLError, TimeoutError, json.JSONDecodeError, ValueError) as error:
            if attempt < 2:
                time.sleep(2**attempt + 0.5)
                continue
            return set(), f"{type(error).__name__}: {error}"
    return set(), "retry limit exhausted"


def main() -> None:
    metadata = get_dataset_metadata(DEFAULT_DATASET)
    split_counts = get_split_counts(metadata)
    annotation_rows: list[dict] = []
    for split, expected_count in sorted(split_counts.items()):
        annotation_rows.extend(
            fetch_split_rows(
                DEFAULT_DATASET,
                "default",
                split,
                DEFAULT_API_ROOT,
                expected_count,
            )
        )

    valid_ids: set[str] = set()
    invalid_count = 0
    for row in annotation_rows:
        normalized, status = normalize_id(row.get("barcode_raw"))
        if status == "valid":
            valid_ids.add(normalized)
        else:
            invalid_count += 1
    codes = sorted(valid_ids)

    found: set[str] = set()
    inconclusive_codes: set[str] = set()
    errors: list[str] = []
    batch_count = (len(codes) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_number, start in enumerate(range(0, len(codes), BATCH_SIZE), start=1):
        batch = codes[start : start + BATCH_SIZE]
        hits, error = get_exact_codes(batch)
        if error is not None:
            inconclusive_codes.update(batch)
            errors.append(f"batch {batch_number}/{batch_count}: {error}")
        else:
            found.update(hits)
        if batch_number < batch_count:
            time.sleep(REQUEST_DELAY_SECONDS)

    not_found = valid_ids - found - inconclusive_codes
    denominator = len(valid_ids)
    found_pct = (100 * len(found) / denominator) if denominator else 0.0
    report_lines = [
        "# Annotated Products in the Complete Open Food Facts Product Index",
        "",
        f"Checked (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "Looked up the unique annotated barcodes against the Open Food Facts Search-a-licious exact-code index in batches. The query requested only `code` fields, so no product images or product datasets were downloaded. Barcodes remain strings; no zero-padding or numeric conversion was applied.",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Annotated products (unique valid barcodes) | {denominator:,} |",
        f"| Products found in the OFF search index | {len(found):,} |",
        f"| Products not found in an exact, complete batch response | {len(not_found):,} |",
        f"| Failed/inconclusive lookups | {len(inconclusive_codes):,} |",
        f"| Overlap percentage | {found_pct:.2f}% |",
        "",
        "## Method and limitations",
        "",
        f"- Hugging Face source: `{DEFAULT_DATASET}`; rows: {len(annotation_rows):,}; splits: {', '.join(f'{name}={count:,}' for name, count in sorted(split_counts.items()))}; invalid/missing IDs excluded: {invalid_count:,}.",
        f"- Exact string search; {BATCH_SIZE} codes per request; {batch_count} sequential requests with {REQUEST_DELAY_SECONDS:.1f}s spacing and bounded retry for transient failures.",
        "- A ‘not found’ result means the current Search-a-licious product index returned an exact, non-timed-out, exact-count batch response without that barcode. Search indexes can lag the underlying product database, so this is not an immutable proof that a product has never existed or is not momentarily absent from the source database.",
        "- Failed/inconclusive batches are excluded from both found and not-found counts. Review them separately; they are not reported as nonmatches.",
        "- No per-product join file, image download, full OFF export, or dashboard/EDA modification was created.",
    ]
    if errors:
        report_lines.extend(["", "## Lookup failures", ""])
        report_lines.extend(f"- {error}" for error in errors)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "annotated_unique_barcodes": denominator,
                "found": len(found),
                "not_found": len(not_found),
                "failed_or_inconclusive": len(inconclusive_codes),
                "overlap_percent": round(found_pct, 4),
                "report": str(REPORT),
                "errors": errors,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
