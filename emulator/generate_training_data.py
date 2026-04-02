"""
Generate training data for JOS-3 emulator.

Runs JOS-3 across a Latin Hypercube sample of input combinations
covering Guatemala's climate range + worker scenarios.
Outputs a CSV of inputs → JOS-3 outputs for training.
"""

import numpy as np
import pandas as pd
import jos3
import time
from scipy.stats import qmc

# --- Input ranges (Guatemala climate + worker scenarios) ---
INPUT_RANGES = {
    "Ta":     (20, 48),      # Air temperature [°C]
    "Tr":     (20, 65),      # Mean radiant temperature [°C]
    "RH":     (15, 100),     # Relative humidity [%]
    "Va":     (0.1, 4.0),    # Air velocity [m/s]
    "PAR":    (1.0, 5.0),    # Physical activity ratio (MET)
    "clo":    (0.1, 1.5),    # Clothing insulation [clo]
    "height": (1.50, 1.90),  # Height [m]
    "weight": (50, 100),     # Weight [kg]
    "age":    (18, 65),       # Age [years]
}

# Simulation duration — long enough for core to respond
SIM_DURATION = 120  # minutes

N_SAMPLES = 20000


def generate_samples(n=N_SAMPLES):
    """Generate Latin Hypercube samples across input ranges."""
    dims = len(INPUT_RANGES)
    sampler = qmc.LatinHypercube(d=dims, seed=42)
    samples = sampler.random(n)

    # Scale to actual ranges
    lower = np.array([v[0] for v in INPUT_RANGES.values()])
    upper = np.array([v[1] for v in INPUT_RANGES.values()])
    scaled = qmc.scale(samples, lower, upper)

    df = pd.DataFrame(scaled, columns=INPUT_RANGES.keys())

    # Add sex as a binary variable (50/50 split)
    rng = np.random.default_rng(42)
    df["sex"] = rng.choice(["male", "female"], size=n)

    # Round integer-like fields
    df["age"] = df["age"].round().astype(int)
    df["RH"] = df["RH"].round().astype(int)

    return df


def run_single(row):
    """Run JOS-3 for a single input combination. Returns dict of outputs."""
    try:
        m = jos3.JOS3(
            height=row["height"],
            weight=row["weight"],
            age=row["age"],
            sex=row["sex"],
        )
        m.Ta = row["Ta"]
        m.Tr = row["Tr"]
        m.RH = row["RH"]
        m.Va = row["Va"]
        m.PAR = row["PAR"]
        m.posture = "standing"
        m.simulate(SIM_DURATION)
        d = m.dict_results()

        # Extract outputs
        tcr = np.array(d["TcrChest"])
        tsk_mean = np.array(d["TskMean"])
        wet_mean = np.array(d["WetMean"])

        peak_core = float(tcr.max())
        final_core = float(tcr[-1])
        peak_skin = float(tsk_mean.max())
        final_skin = float(tsk_mean[-1])
        peak_wet = float(wet_mean.max())
        final_wet = float(wet_mean[-1])

        # Time to core temp thresholds (minutes, -1 if never reached)
        idx_38 = np.where(tcr > 38.0)[0]
        time_to_38 = int(idx_38[0]) if len(idx_38) > 0 else -1

        idx_385 = np.where(tcr > 38.5)[0]
        time_to_385 = int(idx_385[0]) if len(idx_385) > 0 else -1

        idx_39 = np.where(tcr > 39.0)[0]
        time_to_39 = int(idx_39[0]) if len(idx_39) > 0 else -1

        # Core temp rate of change at end (°C/min, last 10 min)
        if len(tcr) > 10:
            core_rate = float((tcr[-1] - tcr[-11]) / 10)
        else:
            core_rate = 0.0

        # Segment skin temps at end
        segments = ["Head", "Chest", "Back", "Pelvis",
                    "LHand", "RHand", "LFoot", "RFoot"]
        seg_temps = {}
        for seg in segments:
            seg_temps[f"Tsk_{seg}"] = float(d[f"Tsk{seg}"][-1])

        return {
            "peak_core": peak_core,
            "final_core": final_core,
            "peak_skin": peak_skin,
            "final_skin": final_skin,
            "peak_wet": peak_wet,
            "final_wet": final_wet,
            "time_to_38": time_to_38,
            "time_to_385": time_to_385,
            "time_to_39": time_to_39,
            "core_rate": core_rate,
            **seg_temps,
        }

    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    print(f"Generating {N_SAMPLES} Latin Hypercube samples...")
    inputs = generate_samples()
    print(f"  Input shape: {inputs.shape}")
    print(f"  Ranges:")
    for col in INPUT_RANGES:
        print(f"    {col}: {inputs[col].min():.1f} – {inputs[col].max():.1f}")

    results = []
    failed = 0
    t0 = time.time()

    for i, row in inputs.iterrows():
        out = run_single(row)
        if out is None:
            failed += 1
            continue
        results.append({**row.to_dict(), **out})

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N_SAMPLES - i - 1) / rate
            print(f"  {i+1}/{N_SAMPLES} done | "
                  f"{rate:.1f} runs/sec | "
                  f"ETA: {eta/3600:.1f}h | "
                  f"failed: {failed}")

    df = pd.DataFrame(results)
    outpath = "emulator/training_data.csv"
    df.to_csv(outpath, index=False)

    elapsed = time.time() - t0
    print(f"\nDone. {len(df)} successful runs in {elapsed/3600:.1f}h")
    print(f"Failed: {failed}")
    print(f"Saved to {outpath}")


if __name__ == "__main__":
    main()
