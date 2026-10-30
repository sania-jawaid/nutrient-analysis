"""Selectively generate BIO overlays and record immutable-cache preservation evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from export_final_dashboard import refresh_bio

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    html = (ROOT / "main_output/index.html").read_text(encoding="utf-8")
    head, rest = html.split('<script id="dashboard2-snapshot" type="application/json">', 1)
    data, tail = rest.split("</script>", 1)
    # Compare everything in the controller except the bounded new renderer.
    if "/* BIO_COMPARISON_START */" in tail:
        a, b = tail.split("/* BIO_COMPARISON_START */", 1)
        _, b = b.split("/* BIO_COMPARISON_END */", 1)
        tail = a + b
    return head, json.loads(data), tail.strip()


def main():
    data_dir = Path(os.environ.get("NUTRITION_DATA_DIR", str(ROOT / "data")))
    protected = [ROOT / "app.py", data_dir / "part2_cache/layoutlm_bio/latest_test_bundle.json",
                 *sorted((ROOT / "outputs/full_cohort_bio").rglob("*.json")),
                 *sorted((data_dir / "part2_cache/images").glob("*")),
                 *sorted((ROOT / "outputs/part1_part2_integration").glob("*.parquet"))]
    protected = [path for path in protected if path.is_file()]
    before_hashes = {str(path): digest(path) for path in protected}
    head, old, tail = snapshot()
    refresh_bio(ROOT)
    new_head, new, new_tail = snapshot()
    assert head == new_head and tail == new_tail, "Custom frozen HTML shell changed"
    retained = [key for key in old if key not in ("bio", "images", "downloads", "source_sha256")]
    for key in retained:
        assert old[key] == new[key], f"Frozen payload changed: {key}"
    for name in ("images", "downloads"):
        assert all(new[name][key] == value for key, value in old[name].items()), name
    after_hashes = {str(path): digest(path) for path in protected}
    assert before_hashes == after_hashes, "Protected source or cache changed"
    examples = new["bio"]["1"]["examples"]
    key = next(iter(examples))
    counts = next(e["html"] for e in examples[key] if "Exact token labels" in e.get("html", ""))
    report = dict(protected_files=len(protected), protected_sha256_unchanged=True,
                  original_app_sha256=before_hashes[str(ROOT / "app.py")],
                  protected_manifest_sha256=hashlib.sha256(json.dumps(before_hashes, sort_keys=True).encode()).hexdigest(),
                  frozen_keys_unchanged=retained, custom_html_shell_unchanged=True,
                  chart_specs=new["chart_spec_count"], html_bytes=(ROOT / "main_output/index.html").stat().st_size,
                  default_seeded_sample=key, default_seeded_counts=counts,
                  embedded_overlay_panels=sum(e["kind"] == "bio_overlay" for source in new["bio"].values()
                                             for events in source["examples"].values() for e in events))
    path = ROOT / "outputs/presentation/bio_visualization_validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()