"""
fetch_dkasc_mirror.py — fetch real DKASC metered data from the GitHub mirror
============================================================================

The official DKASC export backend (solarcentre.spinifexvalley.com.au) has been
returning HTTP 500 for every request. This script fetches the same underlying
Alice Springs metered data from a public GitHub mirror (six-ssp/power), which
hosts raw 5-minute DKASC exports for FOUR arrays of different technologies:

    1A  AliceSprings MonoTrack   (Trina, mono-Si, dual-axis tracker)
    1C  AliceSprings array 1C
    3A  AliceSprings array 3A
    4A  AliceSprings array 4A

Coverage: ~April 2013 – October 2016 (all seasons, 300k+ rows per array).

The mirror's column names differ from the official DKASC export format, so
after download the columns are renamed to the standard names that
``src/dkasc_parser.py`` expects, and the result is written to ``data/``:

    mirror column       →  standard DKASC column
    -------------------    ----------------------------------
    timestamp           →  Timestamp
    temperature         →  Weather_Temperature_Celsius
    humidity            →  Weather_Relative_Humidity
    global_radiation    →  Global_Horizontal_Radiation
    direct_radiation    →  Direct_Normal_Radiation  (kept; parser ignores extras)
    irradiance          →  Plane_of_Array_Irradiance (kept; parser ignores extras)
    wind_speed          →  Wind_Speed
    power               →  {ARRAY}_Active_Power

Usage:
    python fetch_dkasc_mirror.py                # fetch array 1A (default)
    python fetch_dkasc_mirror.py --array 3A     # fetch a different technology
    python fetch_dkasc_mirror.py --all          # fetch all four arrays

Then run the experiment:
    python run_dkasc_experiments.py
"""

from __future__ import annotations

import argparse
import gzip
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

MIRROR_BASE = ("https://raw.githubusercontent.com/six-ssp/power/HEAD/"
               "dataset_parts/data_{array}.csv.gz.part{part:03d}")
ARRAYS = ("1A", "1C", "3A", "4A")
N_PARTS = 3
DATA_DIR = Path(__file__).resolve().parent / "data"

RENAME = {
    "timestamp": "Timestamp",
    "temperature": "Weather_Temperature_Celsius",
    "humidity": "Weather_Relative_Humidity",
    "global_radiation": "Global_Horizontal_Radiation",
    "direct_radiation": "Direct_Normal_Radiation",
    "irradiance": "Plane_of_Array_Irradiance",
    "wind_speed": "Wind_Speed",
}


def fetch_array(array: str) -> Path:
    """Download, reassemble, rename columns, and write data/dkasc_{array}.csv."""
    if array not in ARRAYS:
        raise SystemExit(f"Unknown array {array!r}; choose from {ARRAYS}")

    print(f"[{array}] downloading {N_PARTS} parts from GitHub mirror ...")
    blob = b""
    for part in range(1, N_PARTS + 1):
        url = MIRROR_BASE.format(array=array, part=part)
        with urllib.request.urlopen(url, timeout=300) as resp:
            chunk = resp.read()
        blob += chunk
        print(f"[{array}]   part {part}: {len(chunk):,} bytes")

    raw = gzip.decompress(blob)
    df = pd.read_csv(io.BytesIO(raw))
    print(f"[{array}] parsed {len(df):,} rows "
          f"({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]})")

    df = df.rename(columns=RENAME)
    power_col = f"{array}_Active_Power"
    df[power_col] = pd.to_numeric(df.pop("power"), errors="coerce")
    df = df.drop(columns=["plant_id"], errors="ignore")

    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / f"dkasc_{array}.csv"
    df.to_csv(out, index=False)
    print(f"[{array}] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--array", default="1A", help=f"one of {ARRAYS}")
    ap.add_argument("--all", action="store_true", help="fetch all four arrays")
    args = ap.parse_args()

    targets = ARRAYS if args.all else (args.array,)
    for a in targets:
        fetch_array(a)

    print("\nDone. Next step:")
    print("  python run_dkasc_experiments.py "
          "--csv data/dkasc_1A.csv   (or the array you fetched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
