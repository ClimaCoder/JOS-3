# JOS-3 Neural Net Emulator

First machine learning emulator of the JOS-3 thermoregulation model, enabling gridded heat strain mapping at ERA5-Land scale.

## Why

JOS-3 is a 65-node, 17-segment human thermoregulation model that predicts core temperature, skin temperature, sweating, and blood flow. It's too slow to run at every grid point for climate-scale analysis (~0.5-1s per simulation). This emulator approximates JOS-3 with a small neural network that runs in microseconds.

## Pipeline

```
1. generate_training_data.py    →  training_data.csv (20k JOS-3 runs)
2. train_emulator.py            →  jos3_emulator.pt + scalers.json
3. apply_guatemala.py           →  guatemala_2025_jos3.nc (gridded output)
```

## Inputs (10 features)

| Variable | Range | Unit |
|----------|-------|------|
| Air temperature (Ta) | 20–48 | °C |
| Mean radiant temperature (Tr) | 20–65 | °C |
| Relative humidity (RH) | 15–100 | % |
| Air velocity (Va) | 0.1–4.0 | m/s |
| Physical activity ratio (PAR) | 1.0–5.0 | MET |
| Clothing insulation (clo) | 0.1–1.5 | clo |
| Height | 1.50–1.90 | m |
| Weight | 50–100 | kg |
| Age | 18–65 | years |
| Sex | male/female | binary |

## Outputs (18 variables)

- Peak and final core temperature (°C)
- Peak and final mean skin temperature (°C)
- Peak and final skin wettedness
- Time to core temp 38.0, 38.5, 39.0°C (min)
- Core temperature rate of change (°C/min)
- 8 segment skin temperatures at end of simulation (°C)

## Architecture

Feedforward neural network: 10 → 128 → 128 → 64 → 18 with ReLU + BatchNorm. ~28k parameters.

## Requirements

```
jos3
numpy
pandas
scipy
torch
scikit-learn
xarray
```

## Usage

```bash
# Step 1: Generate training data (~6-8 hours on laptop)
python emulator/generate_training_data.py

# Step 2: Train emulator (~2-5 minutes)
python emulator/train_emulator.py

# Step 3: Apply to Guatemala 2025 (~30 min with ARCO ERA5)
python emulator/apply_guatemala.py
```

## Citation

If you use this emulator, please cite:

- JOS-3: Takahashi et al. (2021), "Thermoregulation model JOS-3 with new open source code", Energy & Buildings, doi:10.1016/j.enbuild.2020.110575
- This emulator: ClimaCoder (2026), github.com/ClimaCoder/JOS-3

## License

MIT (same as JOS-3)
