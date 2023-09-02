import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Parser for model training")

    # Adding command-line arguments
    parser.add_argument("-c", "--conditional", action="store_true", help="Enable conditional forecaster", default=False)
    parser.add_argument("--one-hist", action="store_true", help="Only provide history of one agent", default=False)
    parser.add_argument("--batch-size", type=int, help="Batch size for training", default=64)
    parser.add_argument("--log-dir", type=str, help="Logging directory for tensorboard", default='./logs_default')
    parser.add_argument("--lr-pred", type=float, help="Predictor learning rate", default=3e-4)
    parser.add_argument("--lr-disc", type=float, help="Discriminator learning rate", default=0.0005)
    parser.add_argument("--epochs", type=int, help="Epochs", default=50)

    return parser