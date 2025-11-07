import datetime
import argparse
from daneel.parameters import Parameters
from daneel.detection import transit as transit_mod

def main():
    parser = argparse.ArgumentParser(prog="daneel")

    parser.add_argument(
        "-i", "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input YAML parameters file"
    )

    parser.add_argument(
        "-t", "--transit",
        dest="do_transit",
        action="store_true",
        help="Plot the transit light curve using BATMAN"
    )

    parser.add_argument(
        "-d", "--detect",
        dest="detect",
        action="store_true",
        help="Initialise detection algorithms for Exoplanets (placeholder)"
    )

    parser.add_argument(
        "-a", "--atmosphere",
        dest="complete",
        action="store_true",
        help="Atmospheric Characterization from input transmission spectrum (placeholder)"
    )

    args = parser.parse_args()

    start = datetime.datetime.now()
    print(f"Daneel starts at {start:%Y-%m-%d %H:%M:%S}")

    # Read parameters from YAML (your Parameters class returns a dict-like .params)
    input_pars = Parameters(args.input_file).params

    # ---- Task G: call the transit method when requested ----
    if args.do_transit:
        # Allow overriding output name from YAML if present; otherwise default inside transit()
        out_png = input_pars.get("out_png", None)
        saved = transit_mod.transit(params_yaml=args.input_file, out_png=out_png)
        print("Transit saved to:", saved)

    if args.detect:
        pass  # (placeholder for your future detection code)

    if args.complete:
        pass  # (placeholder for your future atmosphere code)

    finish = datetime.datetime.now()
    print(f"Daneel finishes at {finish:%Y-%m-%d %H:%M:%S}")

if __name__ == "__main__":
    main()


