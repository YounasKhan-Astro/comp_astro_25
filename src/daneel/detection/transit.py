# --- BATMAN tutorial-style transit script for Kepler-297 c ---
# Steps (as in the tutorial flow):
# 1) Define TransitParams
# 2) Create a time array around mid-transit
# 3) Build TransitModel(params, t)
# 4) Compute light curve with m.light_curve(params)
# 5) Plot and save PNG

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import batman
import os

def make_transit_plot(
    # --- Planet/star parameters (Kepler-297 c) ---
    # Period and geometry:
    per=74.92768653,    # days (orbital period)
    inc=89.47,          # deg (inclination)
    ecc=0.0,            # eccentricity
    w=90.0,             # deg (argument of periastron; unused if ecc=0)

    # Scale parameters (assumptions documented):
    # We assume R_star = 1 R_sun → a/R* = a(AU) * 215.032
    a_over_rstar=0.3292 * 215.032,  # ≈ 70.789
    # Rp/R* using Rp≈0.57 R_J and 1 R_J ≈ 0.10045 R_sun:
    rp_over_rstar=0.57 * 0.10045,   # ≈ 0.05726

    # Limb darkening (typical quadratic if not provided):
    u=(0.3, 0.2),
    limb_dark="quadratic",

    # Time window for plotting:
    t_center=0.0,       # mid-transit reference time
    t_window_days=0.5,  # total window width (centered on t_center)

    # Output
    out_png="Kepler-297c_assignment1_taskF.png",
):
    # (1) Transit parameters
    params = batman.TransitParams()
    params.t0 = t_center
    params.per = per
    params.rp = rp_over_rstar
    params.a = a_over_rstar
    params.inc = inc
    params.ecc = ecc
    params.w = w
    params.u = list(u)
    params.limb_dark = limb_dark

    # (2) Time array around mid-transit
    t = np.linspace(t_center - t_window_days/2, t_center + t_window_days/2, 2000)

    # (3) Transit model
    m = batman.TransitModel(params, t)

    # (4) Model light curve
    flux = m.light_curve(params)

    # (5) Plot and save
    plt.figure()
    plt.plot(t, flux, lw=2)
    plt.xlabel("Time from mid-transit [days]")
    plt.ylabel("Relative flux")
    plt.title("Kepler-297 c — Transit Light Curve (BATMAN)")
    plt.tight_layout()
    out_path = os.path.join(os.getcwd(), out_png)
    plt.savefig(out_path, dpi=150)
    print("Saved:", out_path)
    return out_path

if __name__ == "__main__":
    make_transit_plot()
 
