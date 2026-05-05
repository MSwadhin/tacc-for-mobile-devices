#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import ssl
import textwrap
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


OFFICIAL_SOURCES = [
    "https://www.cs.dartmouth.edu/~dfk/research/kotz-dartmouth-campus-20090909/index.html",
    "https://doi.org/10.15783/C7F59T",
    "https://ieee-dataport.org/open-access/crawdad-dartmouthcampus-v-2004-11-09",
]


def write_access_note(path: Path) -> None:
    note = f"""\
    Dartmouth/CRAWDAD campus dataset access note
    ===========================================

    The current project uses the Dartmouth campus WLAN mobility trace as the intended
    real-world source for AP handoff topology construction. The public Dartmouth
    author page now points readers to IEEE DataPort. Legacy direct CRAWDAD download
    paths checked during implementation returned HTTP 404. IEEE DataPort provides the
    dataset landing page, but programmatic download may require browser interaction,
    account access, or DataPort terms acceptance.

    Official sources checked:
    {chr(10).join("- " + url for url in OFFICIAL_SOURCES)}

    Reproducibility fallback:
    When data/raw/dartmouth_movement.csv is absent, scripts/run_experiments.py creates
    a deterministic Dartmouth-like synthetic AP movement trace with columns:
    user_id,timestamp,ap. This fallback preserves the project pipeline and is clearly
    marked in results/run_metadata.json under used_synthetic_fallback.

    To use a manually downloaded real trace, convert it to CSV with columns
    user_id,timestamp,ap and save it as data/raw/dartmouth_movement.csv, or pass its
    path with scripts/run_experiments.py --trace.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(note), encoding="utf-8")


def try_download_landing_pages(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ssl_context = ssl._create_unverified_context()
    for idx, url in enumerate(OFFICIAL_SOURCES, start=1):
        target = output_dir / f"source_{idx}.html"
        try:
            with urllib.request.urlopen(url, timeout=20, context=ssl_context) as response:
                target.write_bytes(response.read())
                print(f"Saved landing page: {target}")
        except Exception as exc:
            print(f"Could not fetch {url}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Dartmouth dataset access information.")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "raw" / "dartmouth_access"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    write_access_note(output_dir / "ACCESS_NOTE.txt")
    try_download_landing_pages(output_dir)
    print(f"Wrote dataset access note to {output_dir / 'ACCESS_NOTE.txt'}")


if __name__ == "__main__":
    main()
