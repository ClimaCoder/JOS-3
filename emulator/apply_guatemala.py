"""
Apply trained JOS-3 emulator to ERA5-Land data for Guatemala (2025).

Pulls daily worst-case conditions from ARCO ERA5, runs the emulator
on every grid point, and saves gridded heat strain maps.
"""

import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import json
import time

# Guatemala bounding box
LAT_MIN, LAT_MAX = 13.5, 18.0
LON_MIN, LON_MAX = -92.5, -88.0

# Default worker profile (can be parameterized)
WORKER_DEFAULTS = {
    "PAR": 3.0,       # moderate-heavy labor
    "clo": 0.5,       # light work clothing
    "height": 1.65,   # average Guatemalan male
    "weight": 68,
    "age": 35,
    "sex_male": 1.0,
}

ERA5_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


class JOS3Emulator(nn.Module):
    def __init__(self, n_in, n_out, hidden_sizes):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_emulator():
    """Load trained model and scalers."""
    with open("emulator/scalers.json") as f:
        meta = json.load(f)

    model = JOS3Emulator(
        n_in=len(meta["input_cols"]),
        n_out=len(meta["output_cols"]),
        hidden_sizes=meta["hidden_sizes"],
    )
    model.load_state_dict(torch.load("emulator/jos3_emulator.pt",
                                     weights_only=True))
    model.eval()

    return model, meta


def predict(model, meta, inputs_df):
    """Run emulator on a DataFrame of inputs. Returns DataFrame of outputs."""
    X = inputs_df[meta["input_cols"]].values.astype(np.float32)

    # Scale
    x_mean = np.array(meta["x_mean"])
    x_scale = np.array(meta["x_scale"])
    X_s = (X - x_mean) / x_scale

    # Predict
    with torch.no_grad():
        Y_s = model(torch.tensor(X_s)).numpy()

    # Unscale
    y_mean = np.array(meta["y_mean"])
    y_scale = np.array(meta["y_scale"])
    Y = Y_s * y_scale + y_mean

    return pd.DataFrame(Y, columns=meta["output_cols"], index=inputs_df.index)


def load_era5_day(ds, date):
    """Extract daily worst-case conditions for Guatemala from ERA5."""
    from mrt import compute_mrt

    day = ds.sel(time=date)

    # Select Guatemala region
    # ERA5 latitude is descending (90 to -90)
    day = day.sel(
        latitude=slice(LAT_MAX, LAT_MIN),
        longitude=slice(LON_MIN, LON_MAX),
    )

    # Find the hottest hour by air temperature
    t2m_hourly = day["2m_temperature"] - 273.15
    hottest_hour = t2m_hourly.mean(dim=["latitude", "longitude"]).argmax().item()

    # Extract variables at hottest hour for worst-case analysis
    peak = day.isel(time=hottest_hour)

    # Air temperature [°C]
    t2m = peak["2m_temperature"] - 273.15

    # Radiation variables for MRT (accumulated over 1h step)
    ssrd = peak["surface_solar_radiation_downwards"]
    ssr = peak["surface_net_solar_radiation"]
    strd = peak["surface_thermal_radiation_downwards"]
    strn = peak["surface_net_thermal_radiation"]
    fal = peak["forecast_albedo"]

    # MRT from full radiation budget (Di Napoli et al. 2020)
    tr = compute_mrt(ssrd.values, ssr.values, strd.values, strn.values, fal.values)
    tr = xr.DataArray(tr, coords=t2m.coords)

    # Dewpoint → RH at hottest hour
    d2m = peak["2m_dewpoint_temperature"] - 273.15
    # Magnus formula for RH from Ta and Td
    rh = 100 * np.exp(17.625 * d2m / (243.04 + d2m)) / \
              np.exp(17.625 * t2m / (243.04 + t2m))
    rh = rh.clip(10, 100)

    # Wind speed at hottest hour
    u10 = peak["10m_u_component_of_wind"]
    v10 = peak["10m_v_component_of_wind"]
    va = np.sqrt(u10**2 + v10**2)
    va = va.clip(0.1, None)  # JOS-3 needs > 0

    return t2m, tr, rh, va


def process_day(model, meta, ds, date):
    """Run emulator for one day across all Guatemala grid points."""
    t2m, tr, rh, va = load_era5_day(ds, date)

    lats = t2m.latitude.values
    lons = t2m.longitude.values
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    # Flatten grids
    n_points = lat_grid.size
    inputs = pd.DataFrame({
        "Ta": t2m.values.flatten(),
        "Tr": tr.values.flatten(),
        "RH": rh.values.flatten(),
        "Va": va.values.flatten(),
        "PAR": WORKER_DEFAULTS["PAR"],
        "clo": WORKER_DEFAULTS["clo"],
        "height": WORKER_DEFAULTS["height"],
        "weight": WORKER_DEFAULTS["weight"],
        "age": WORKER_DEFAULTS["age"],
        "sex_male": WORKER_DEFAULTS["sex_male"],
    })

    # Run emulator
    outputs = predict(model, meta, inputs)

    # Reshape back to grid
    result = {}
    for col in outputs.columns:
        result[col] = outputs[col].values.reshape(len(lats), len(lons))

    return xr.Dataset(
        {k: (["latitude", "longitude"], v) for k, v in result.items()},
        coords={"latitude": lats, "longitude": lons},
    ).expand_dims(time=[pd.Timestamp(date)])


def main():
    print("Loading emulator...")
    model, meta = load_emulator()

    print("Opening ERA5 dataset...")
    ds = xr.open_zarr(
        ERA5_ZARR,
        chunks=None,
        storage_options={"token": "anon"},
    )

    # Process 2025
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    all_days = []

    t0 = time.time()
    for i, date in enumerate(dates):
        date_str = date.strftime("%Y-%m-%d")
        try:
            day_result = process_day(model, meta, ds, date_str)
            all_days.append(day_result)
        except Exception as e:
            print(f"  {date_str} failed: {e}")
            continue

        if (i + 1) % 30 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60
            print(f"  {i+1}/365 days | {rate:.0f} days/min | "
                  f"{date_str}")

    # Combine and save
    print("Combining results...")
    combined = xr.concat(all_days, dim="time")

    outpath = "emulator/guatemala_2025_jos3.nc"
    combined.to_netcdf(outpath)
    print(f"\nSaved to {outpath}")
    print(f"Shape: {combined.dims}")
    print(f"Variables: {list(combined.data_vars)}")
    print(f"Total time: {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
