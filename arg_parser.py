import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Parser for model training")

    # Adding command-line arguments
    parser.add_argument("-c", "--conditional", action="store_true", help="Enable conditional forecaster", default=True)
    parser.add_argument("--one-hist", action="store_true", help="Only provide history of one agent", default=False)

    return parser