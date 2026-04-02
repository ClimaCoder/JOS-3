"""
Mean Radiant Temperature (MRT) from ERA5 radiation variables.

Based on Di Napoli et al. (2020):
  "Mean radiant temperature from global-scale numerical weather prediction models"
  Int J Biometeorol, doi:10.1007/s00484-020-01900-5

And ERA5-HEAT (Di Napoli et al. 2021):
  "ERA5-HEAT: A global gridded historical dataset of human thermal comfort indices"
  Geoscience Data Journal, doi:10.1002/gdj3.102

Uses the two-hemisphere approach: radiation fluxes split into
sky (downward) and ground (upward) components, with angular
weighting factors for a standing person.

ERA5 variables needed:
  - ssrd: Surface solar radiation downwards [J/m²] (accumulated)
  - ssr:  Surface net solar radiation [J/m²] (accumulated)
  - strd: Surface thermal radiation downwards [J/m²] (accumulated)
  - str:  Surface net thermal radiation [J/m²] (accumulated)
  - fal:  Forecast albedo [-]

All radiation variables in ERA5 are accumulated over the forecast step.
Divide by step duration (3600s for hourly) to get W/m².
"""

import numpy as np

# Stefan-Boltzmann constant [W/m²/K⁴]
SIGMA = 5.67e-8

# Emissivity of the human body [-]
EPSILON_P = 0.97

# Absorption coefficient for shortwave radiation [-]
# ~0.7 for average clothing/skin (ISO 7726)
ALPHA_SW = 0.7

# Angular factors for a standing person (two-hemisphere model)
# fp_up = fp_down = 0.06 (fraction of body area facing up/down)
# fp_side = 0.22 (fraction facing each horizontal direction)
# For two-hemisphere: f_sky = f_ground = 0.5
# But standing person projects differently:
#   - sees more from sides than from directly above/below
#   - Fanger (1972): fp for standing person to horizontal surface ≈ 0.06
#   - fp for standing person to vertical half-space ≈ 0.22
# Di Napoli simplified approach: equal hemispheric weighting
F_SKY = 0.5    # angular factor for upper hemisphere
F_GROUND = 0.5  # angular factor for lower hemisphere


def era5_radiation_to_flux(accumulated_values, step_seconds=3600):
    """
    Convert ERA5 accumulated radiation [J/m²] to flux [W/m²].

    ERA5 radiation variables are accumulated over the forecast step.
    For hourly data, divide by 3600 to get mean flux in W/m².
    """
    return accumulated_values / step_seconds


def compute_mrt(ssrd, ssr, strd, strn, fal, step_seconds=3600):
    """
    Compute Mean Radiant Temperature from ERA5 radiation variables.

    Parameters
    ----------
    ssrd : array
        Surface solar radiation downwards [J/m²] (accumulated)
    ssr : array
        Surface net solar radiation [J/m²] (accumulated)
    strd : array
        Surface thermal radiation downwards [J/m²] (accumulated)
    strn : array
        Surface net thermal radiation [J/m²] (accumulated)
        Note: ERA5 variable name is 'str' but that's a Python keyword
    fal : array
        Forecast albedo [-] (0-1)
    step_seconds : float
        Accumulation period in seconds (3600 for hourly ERA5)

    Returns
    -------
    mrt : array
        Mean Radiant Temperature [°C]
    """
    # Convert accumulated values to fluxes [W/m²]
    ssrd_f = era5_radiation_to_flux(ssrd, step_seconds)
    ssr_f = era5_radiation_to_flux(ssr, step_seconds)
    strd_f = era5_radiation_to_flux(strd, step_seconds)
    strn_f = era5_radiation_to_flux(strn, step_seconds)

    # --- Shortwave (solar) radiation ---
    # Downward SW from sky
    sw_down = ssrd_f  # [W/m²]

    # Upward SW from ground (reflected)
    # ssr = ssrd - reflected, so reflected = ssrd - ssr
    # OR: reflected = ssrd * albedo
    sw_up = ssrd_f * fal  # [W/m²]

    # Total shortwave absorbed by standing person
    # Using isotropic two-hemisphere model
    sw_absorbed = ALPHA_SW * (F_SKY * sw_down + F_GROUND * sw_up)

    # --- Longwave (thermal) radiation ---
    # Downward LW from sky (atmosphere)
    lw_down = strd_f  # [W/m²]

    # Upward LW from ground
    # str (net) = strd - upward_LW, so upward_LW = strd - str
    lw_up = strd_f - strn_f  # [W/m²]

    # Total longwave absorbed by standing person
    lw_absorbed = EPSILON_P * (F_SKY * lw_down + F_GROUND * lw_up)

    # --- Mean Radiant Temperature ---
    # Total absorbed radiation per unit body area
    r_total = sw_absorbed + lw_absorbed

    # MRT from Stefan-Boltzmann:  R_total = epsilon * sigma * Tmrt^4
    # Tmrt = (R_total / (epsilon * sigma))^0.25
    mrt_k = (r_total / (EPSILON_P * SIGMA)) ** 0.25  # [K]
    mrt_c = mrt_k - 273.15  # [°C]

    return mrt_c


def compute_mrt_simple(t2m_c, ssrd, step_seconds=3600):
    """
    Simplified MRT when only air temp and downward solar are available.

    Less accurate than full method but requires fewer variables.
    Based on empirical relationship from Kantor & Unger (2011).

    Parameters
    ----------
    t2m_c : array
        2m air temperature [°C]
    ssrd : array
        Surface solar radiation downwards [J/m²] (accumulated)
    step_seconds : float
        Accumulation period in seconds

    Returns
    -------
    mrt : array
        Estimated Mean Radiant Temperature [°C]
    """
    ssrd_f = era5_radiation_to_flux(ssrd, step_seconds)

    # Empirical: MRT ≈ Ta + coefficient * solar_flux
    # Coefficient derived from Kantor & Unger (2011) for open areas
    # At 800 W/m² solar, MRT is typically 15-25°C above Ta
    # Using ~0.025 °C per W/m² as a middle estimate
    mrt = t2m_c + 0.025 * ssrd_f

    return mrt
