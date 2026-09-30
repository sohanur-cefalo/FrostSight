"""One-off data cleaning: strips the route prefix out of an already-generated
segments.csv's road_name column.

Older prototype/build_dataset.py runs wrote road_name as "<route> · <place>"
(e.g. "F7940 · Arnøya") — redundant, since route_code already carries the
route separately. This rewrites road_name to just the place name ("Arnøya"),
in place, for a segments.csv you already have on disk (build_dataset.py itself
is fixed to not do this combining any more, so a fresh run doesn't need this).

Run: python -m prototype.clean_road_names
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("prototype.clean_road_names")

SEGMENTS_PATH = "prototype/artifacts/segments.csv"
SEPARATOR = " · "


def clean_road_names(segments: pd.DataFrame) -> pd.DataFrame:
    """road_name "F7940 · Arnøya" -> "Arnøya". A road_name with no separator
    (already clean, or synthetic data's "E8/003" style) is left untouched."""
    segments = segments.copy()
    has_prefix = segments["road_name"].str.contains(SEPARATOR, regex=False)
    segments.loc[has_prefix, "road_name"] = (
        segments.loc[has_prefix, "road_name"].str.split(SEPARATOR, n=1).str[1]
    )
    return segments


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    segments = pd.read_csv(SEGMENTS_PATH)
    before = segments["road_name"].copy()

    cleaned = clean_road_names(segments)
    n_changed = (cleaned["road_name"] != before).sum()

    cleaned.to_csv(SEGMENTS_PATH, index=False)
    log.info("Cleaned road_name on %d of %d rows in %s", n_changed, len(cleaned), SEGMENTS_PATH)


if __name__ == "__main__":
    main()
