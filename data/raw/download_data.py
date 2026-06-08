"""
download_data.py - Download real public datasets for Zimbabwe cholera forecasting.
Sources: HDX, ERA5 (CDS API), World Bank Open Data

Usage:
    python data/raw/download_data.py --source all
    python data/raw/download_data.py --source worldbank
"""

import argparse, logging, requests
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_world_bank(indicators=None):
    """Download World Bank WASH and poverty indicators for Zimbabwe."""
    if indicators is None:
        indicators = ["SH.H2O.BASW.ZS", "SH.STA.BASS.ZS", "SP.POP.TOTL", "SI.POV.NAHC"]
    all_data = []
    for ind in indicators:
        url = f"https://api.worldbank.org/v2/country/ZW/indicator/{ind}?format=json&per_page=50&mrv=10"
        logger.info(f"Fetching: {ind}")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if len(data) > 1 and data[1]:
                for entry in data[1]:
                    if entry.get("value") is not None:
                        all_data.append({"indicator": ind, "year": int(entry["date"]),
                                         "value": float(entry["value"])})
        except Exception as e:
            logger.error(f"Failed {ind}: {e}")
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(OUTPUT_DIR / "world_bank_indicators.csv", index=False)
        logger.info(f"Saved {len(df)} rows -> data/raw/world_bank_indicators.csv")
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


def hdx_instructions():
    logger.info(
        "\n[HDX] Download Zimbabwe cholera data manually:\n"
        "  1. https://data.humdata.org/search?q=zimbabwe+cholera\n"
        "  2. Save to: data/raw/hdx_cholera.csv\n"
    )


def era5_instructions():
    logger.info(
        "\n[ERA5] Setup CDS API: https://cds.climate.copernicus.eu/api-how-to\n"
        "  pip install cdsapi\n"
        "  Then run: python data/raw/era5_download.py\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["all", "worldbank", "hdx", "era5"], default="worldbank")
    args = parser.parse_args()
    if args.source in ("all", "worldbank"):
        download_world_bank()
    if args.source in ("all", "hdx"):
        hdx_instructions()
    if args.source in ("all", "era5"):
        era5_instructions()
    logger.info("\nFalling back to synthetic data is always available:\n  python data/synthetic/generate_synthetic.py\n")

if __name__ == "__main__":
    main()
