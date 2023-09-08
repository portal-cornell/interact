import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Parser for model training")

    # Adding command-line arguments
    parser.add_argument("-c", "--conditional", action="store_true", help="Enable conditional forecaster", default=False)
    parser.add_argument("--one-hist", action="store_true", help="Only provide history of one agent", default=False)
    parser.add_argument("--bob-hand", action="store_true", help="Train on bob wrist/hand only", default=False)
    parser.add_argument("--no-amass", action="store_true", help="Don't train on AMASS", default=False)
    parser.add_argument("--condition-last", action="store_true", help="Forecast conditioned only on the last step", default=False)
    parser.add_argument("--batch-size", type=int, help="Batch size for training", default=256)
    parser.add_argument("--log-dir", type=str, help="Logging directory for tensorboard", default='./logs_default')
    parser.add_argument("--lr-pred", type=float, help="Predictor learning rate", default=3e-4)
    parser.add_argument("--lr-ft", type=float, help="Predictor learning rate for finetuning", default=1e-4)
    parser.add_argument("--lr-disc", type=float, help="Discriminator learning rate", default=0.0005)
    parser.add_argument("--epochs", type=int, help="Epochs", default=50)
    parser.add_argument("--eval", type=str, help="Evaluation dataset", default='cmu', choices=['cmu','amass'])

    return parser