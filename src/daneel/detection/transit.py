import numpy as np
import batman
import matplotlib.pyplot as plt
import yaml 

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
    return params_dict


def transit_from_yaml(yaml_path, output_file="transit_cli.png"):
    model = TransitModel(params_dict)
    model.run(output_file=output_file)


if __name__ == "__main__":
    # Build parameter dictionary for Kepler-297 c
    params = kepler_297c_params()

    # Create model and run it, saving to the required filename
    model = TransitModel(params)
    model.run(output_file="Kepler-297c_assignment1_taskF.png")
