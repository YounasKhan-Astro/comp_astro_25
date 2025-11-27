import datetime
import argparse
from daneel.parameters import Parameters
from daneel.detection.transit import transit as transit_cli


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input parameter file",
    )

    parser.add_argument(
        "-t",
        "--transit",
        dest="transit",
        action="store_true",
        help="Plots transit lightcurve from YAML file",
    )

    parser.add_argument(
        "-d",
        "--detect",
        dest="detect",
        action="store_true",
        help="Initialise detection algorithms for exoplanets",
    )

    parser.add_argument(
        "-a",
        "--atmosphere",
        dest="atmosphere",
        action="store_true",
        help="Atmospheric characterisation",
    )

    # This must be inside main()
    args = parser.parse_args()

    # Start log
    start = datetime.datetime.now()
    print(f"Daneel starts at {start}")

    # ===========================
    # Transit mode (Task G + H)
    # ===========================
    if args.transit:
        transit_cli(params_yaml=args.input_file)

    # ===========================
    # Other modes (unused now)
    # ===========================
    elif args.detect or args.atmosphere:
        input_pars = Parameters(args.input_file).params

        if args.detect:
            pass
        elif args.atmosphere:
            pass

    # End log
    finish = datetime.datetime.now()
    print(f"Daneel finishes at {finish}")


if __name__ == "__main__":
    main()
