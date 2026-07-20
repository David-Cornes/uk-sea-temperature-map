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
    {"name": "Dunbar", "lat": 56.0027, "lon": -2.5167, "sample_lat": 56.00, "sample_lon": -2.45},
    {"name": "Lossiemouth", "lat": 57.7219, "lon": -3.2834, "sample_lat": 57.75, "sample_lon": -3.25},
    {"name": "Portree", "lat": 57.4125, "lon": -6.1942, "sample_lat": 57.45, "sample_lon": -6.25},
    {"name": "Outer Hebrides", "lat": 57.7600, "lon": -7.0200, "sample_lat": 57.80, "sample_lon": -7.20},
    {"name": "Shetland", "lat": 60.1540, "lon": -1.1490, "sample_lat": 60.20, "sample_lon": -1.10},
    {"name": "Orkney", "lat": 58.9847, "lon": -2.9590, "sample_lat": 59.05, "sample_lon": -2.95},
    {"name": "Dublin", "lat": 53.3498, "lon": -6.2603, "sample_lat": 53.35, "sample_lon": -6.05},
    {"name": "Galway", "lat": 53.2707, "lon": -9.0568, "sample_lat": 53.25, "sample_lon": -9.20},
    {"name": "Cork", "lat": 51.8985, "lon": -8.4756, "sample_lat": 51.80, "sample_lon": -8.30},
    {"name": "Arran", "lat": 55.5810, "lon": -5.2080, "sample_lat": 55.55, "sample_lon": -5.35},
    {"name": "Isle of Man", "lat": 54.2361, "lon": -4.5481, "sample_lat": 54.20, "sample_lon": -4.65},
    {"name": "Whitehaven", "lat": 54.5489, "lon": -3.5841, "sample_lat": 54.55, "sample_lon": -3.70},
    {"name": "Barrow-in-Furness", "lat": 54.1109, "lon": -3.2276, "sample_lat": 54.05, "sample_lon": -3.25},
    {"name": "Rhyl", "lat": 53.3191, "lon": -3.4916, "sample_lat": 53.35, "sample_lon": -3.50},
    {"name": "Colwyn Bay", "lat": 53.2948, "lon": -3.7276, "sample_lat": 53.33, "sample_lon": -3.72},
    {"name": "Holyhead", "lat": 53.3106, "lon": -4.6330, "sample_lat": 53.35, "sample_lon": -4.70},
    {"name": "Aberystwyth", "lat": 52.4153, "lon": -4.0829, "sample_lat": 52.42, "sample_lon": -4.20},
    {"name": "Fishguard", "lat": 51.9936, "lon": -4.9769, "sample_lat": 52.00, "sample_lon": -5.05},
    {"name": "Southend-on-Sea", "lat": 51.5459, "lon": 0.7077, "sample_lat": 51.52, "sample_lon": 0.85},
    {"name": "Great Yarmouth", "lat": 52.6083, "lon": 1.7305, "sample_lat": 52.62, "sample_lon": 1.85},
    {"name": "Wells-next-the-Sea", "lat": 52.9543, "lon": 0.8513, "sample_lat": 53.02, "sample_lon": 0.90},
    {"name": "Grimsby", "lat": 53.5675, "lon": -0.0808, "sample_lat": 53.58, "sample_lon": 0.05},
    {"name": "Hugh Town", "lat": 49.9146, "lon": -6.3143, "sample_lat": 49.90, "sample_lon": -6.35},
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

def get_nearest_valid_sst_c(sst, latitude, longitude, search_radius=3):
    """
    Try the nearest grid cell first.
    If it is land/masked/null, search nearby grid cells and return the first valid sea temperature.
    """

    lat_values = sst["latitude"].values
    lon_values = sst["longitude"].values

    lat_index = int(np.abs(lat_values - latitude).argmin())
    lon_index = int(np.abs(lon_values - longitude).argmin())

    # Search nearby cells in expanding squares
    for radius in range(search_radius + 1):
        for i in range(lat_index - radius, lat_index + radius + 1):
            for j in range(lon_index - radius, lon_index + radius + 1):
                if i < 0 or j < 0:
                    continue
                if i >= len(lat_values) or j >= len(lon_values):
                    continue

                value_k = sst.isel(latitude=i, longitude=j).item()

                if value_k is not None and not np.isnan(value_k):
                    return round(float(value_k) - 273.15, 1), float(lat_values[i]), float(lon_values[j])

    return None, None, None
    
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

        temp_c, actual_sample_lat, actual_sample_lon = get_nearest_valid_sst_c(
            sst,
            sample_lat,
            sample_lon,
            search_radius=5
        )

        if temp_c is None:
            advice = None
        else:
            advice = wetsuit_advice(temp_c)

        results.append({
            "location": loc["name"],
            "lat": loc["lat"],
            "lon": loc["lon"],
            "sample_lat": actual_sample_lat if actual_sample_lat is not None else sample_lat,
            "sample_lon": actual_sample_lon if actual_sample_lon is not None else sample_lon,
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
