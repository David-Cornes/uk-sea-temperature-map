import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr


INPUT_FILE = Path("data/uk_ostia_sst.nc")
OUTPUT_FILE = Path("docs/uk-water-temperatures.json")

LOCATIONS = [
    {"name": "Bournemouth", "lat": 50.7192, "lon": -1.8808, "sample_lat": 50.66, "sample_lon": -1.88},
    {"name": "Brighton", "lat": 50.8225, "lon": -0.1372, "sample_lat": 50.76, "sample_lon": -0.13},
    {"name": "Plymouth", "lat": 50.3755, "lon": -4.1427, "sample_lat": 50.30, "sample_lon": -4.10},
    {"name": "Newquay", "lat": 50.4155, "lon": -5.0737},
    {"name": "Croyde", "lat": 51.1300, "lon": -4.2350},
    {"name": "St Ives", "lat": 50.2140, "lon": -5.4800},
    {"name": "Bude", "lat": 50.8280, "lon": -4.5440, "sample_lat": 50.84, "sample_lon": -4.62},
    {"name": "Swansea", "lat": 51.6214, "lon": -3.9436, "sample_lat": 51.55, "sample_lon": -3.98},
    {"name": "Pembrokeshire", "lat": 51.6726, "lon": -5.0340},
    {"name": "Scarborough", "lat": 54.2831, "lon": -0.3998},
    {"name": "Tynemouth", "lat": 55.0170, "lon": -1.4230},
    {"name": "Aberdeen", "lat": 57.1497, "lon": -2.0943},
    {"name": "Thurso", "lat": 58.5933, "lon": -3.5221},
    {"name": "Oban", "lat": 56.4154, "lon": -5.4718},
    {"name": "Portrush", "lat": 55.2047, "lon": -6.6527},
]


def wetsuit_advice(temp_c: float) -> dict:
    if temp_c < 8:
        return {
            "label": "Very cold",
            "advice": "5mm or 6mm winter wetsuit, boots, gloves and hood recommended.",
            "collection": "/collections/winter-wetsuits",
        }
    if temp_c < 11:
        return {
            "label": "Cold",
            "advice": "5mm winter wetsuit with boots, gloves and hood recommended.",
            "collection": "/collections/5mm-wetsuits",
        }
    if temp_c < 14:
        return {
            "label": "Cool",
            "advice": "4/3mm or 5/4mm wetsuit recommended, depending on session length.",
            "collection": "/collections/4mm-wetsuits",
        }
    if temp_c < 17:
        return {
            "label": "Mild",
            "advice": "3/2mm full wetsuit is usually suitable.",
            "collection": "/collections/3mm-wetsuits",
        }
    if temp_c < 20:
        return {
            "label": "Warm",
            "advice": "3/2mm, shorty wetsuit or wetsuit top depending on conditions.",
            "collection": "/collections/summer-wetsuits",
        }
    return {
        "label": "Very warm",
        "advice": "Shorty wetsuit, rash vest or wetsuit top may be enough.",
        "collection": "/collections/rash-vests",
    }


def main():
    ds = xr.open_dataset(INPUT_FILE)
    sst = ds["analysed_sst"]

    data_time = None

    if "time" in sst.dims:
        data_time = str(ds["time"].values[-1])
        sst = sst.isel(time=-1)

    results = []

    for loc in LOCATIONS:
        sample_lat = loc.get("sample_lat", loc["lat"])
        sample_lon = loc.get("sample_lon", loc["lon"])

        value_k = sst.sel(
            latitude=sample_lat,
            longitude=sample_lon,
            method="nearest"
        ).item()

        if value_k is None or np.isnan(value_k):
            temp_c = None
            advice = None
        else:
            temp_c = round(float(value_k) - 273.15, 1)
            advice = wetsuit_advice(temp_c)

        results.append({
            "location": loc["name"],
            "lat": loc["lat"],
            "lon": loc["lon"],
            "sample_lat": sample_lat,
            "sample_lon": sample_lon,
            "temperature_c": temp_c,
            "advice": advice,
        })

    output = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "data_time_utc": data_time,
        "source": "Copernicus Marine OSTIA / Met Office",
        "product_id": "SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001",
        "dataset_id": "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
        "note": "Daily sea surface temperature estimates. Local beach readings may vary.",
        "locations": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
