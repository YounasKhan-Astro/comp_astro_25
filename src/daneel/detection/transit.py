import numpy as np
import batman
import matplotlib.pyplot as plt


class TransitModel:
    """
    Transit model class for exoplanet detection using batman.

    This class generates synthetic light curves for exoplanet transits
    and provides visualization capabilities.
    """

    def __init__(self, params_dict):
        """
        Initialize the transit model with parameters from configuration file.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing transit parameters
        """
        # Batman transit parameters object
        self.params = batman.TransitParams()

        # Basic orbital / geometry parameters
        self.params.t0 = params_dict.get("t0", 0.0)       # mid-transit time [days]
        self.params.per = params_dict.get("per")          # orbital period [days]
        self.params.rp = params_dict.get("rp")            # Rp/R*
        self.params.a = params_dict.get("a")              # a/R*
        self.params.inc = params_dict.get("inc")          # inclination [deg]
        self.params.ecc = params_dict.get("ecc", 0.0)     # eccentricity
        self.params.w = params_dict.get("w", 90.0)        # arg of periastron [deg]

        # Limb darkening
        self.params.u = params_dict.get("u", [0.3, 0.2])  # limb darkening coeffs
        self.params.limb_dark = params_dict.get(
            "limb_dark", "quadratic"
        )  # limb darkening law

        # Time array around mid-transit (in days)
        # Here we look at +/- 0.75 days around t0 with 2000 points
        self.t = np.linspace(-0.75, 0.75, 2000)

        # Initialize batman model
        self.model = batman.TransitModel(self.params, self.t)

    def compute_light_curve(self):
        """
        Compute the light curve using the batman transit model.

        Returns
        -------
        flux : ndarray
            Array of relative flux values
        """
        self.flux = self.model.light_curve(self.params)
        return self.flux

    def plot_light_curve(self, output_file="lc.png"):
        """
        Plot and save the transit light curve.

        Parameters
        ----------
        output_file : str, optional
            Output filename for the plot (default: 'lc.png')
        """
        if not hasattr(self, "flux"):
            self.compute_light_curve()

        plt.figure(figsize=(10, 6))
        plt.plot(self.t, self.flux)
        plt.xlabel("Time from central transit (days)")
        plt.ylabel("Relative flux")
        plt.title("Transit light curve")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()
        print(f"Light curve saved to {output_file}")

    def run(self, output_file="lc.png"):
        """
        Run the complete transit model workflow: compute and plot light curve.
        """
        self.compute_light_curve()
        self.plot_light_curve(output_file)


def kepler_297c_params():
    """
    Parameters for Kepler-297 c, based on exoplanet.eu values.

    Website values (approx):
        Period P          = 74.92768653 days
        Semi-major axis a = 0.3292 AU
        Planet radius     = 0.57 R_J
        Stellar radius    = 0.89 R_sun
        Inclination       = 89.47 deg
        Eccentricity      = 0.0
    We convert a and Rp into units of stellar radii (a/R*, Rp/R*).
    """
    # Given values from the exoplanet catalogue
    P_days = 74.92768653
    a_AU = 0.3292
    Rp_Rj = 0.57
    R_star_Rsun = 0.89

    # Conversion factors
    RJ_OVER_RSUN = 0.10045   # 1 R_J in units of R_sun
    AU_OVER_RSUN = 215.032   # 1 AU in units of R_sun

    # Dimensionless ratios needed by batman
    rp_over_rstar = Rp_Rj * RJ_OVER_RSUN / R_star_Rsun
    a_over_rstar = a_AU * AU_OVER_RSUN / R_star_Rsun

    params_dict = {
        "t0": 0.0,                     # choose reference mid-transit time
        "per": P_days,                 # orbital period [days]
        "rp": rp_over_rstar,           # Rp/R*
        "a": a_over_rstar,             # a/R*
        "inc": 89.47,                  # inclination [deg]
        "ecc": 0.0,                    # eccentricity
        "w": 90.0,                     # argument of periastron [deg]
        "u": [0.3, 0.2],               # limb darkening coefficients
        "limb_dark": "quadratic",      # limb darkening law
    }

    return params_dict


if __name__ == "__main__":
    # Build parameter dictionary for Kepler-297 c
    params = kepler_297c_params()

    # Create model and run it, saving to the required filename
    model = TransitModel(params)
    model.run(output_file="Kepler-297c_assignment1_taskF.png")
