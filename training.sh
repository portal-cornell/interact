combinations=(
    "-c --bob-hand"
    "-c"
    "--bob-hand"
    ""
    "--one-hist -c --bob-hand"
    "--one-hist -c"
    "--one-hist --bob-hand"
    "--one-hist"
)

for args in "${combinations[@]}"; do
    python train_cond_forecaster.py --log-dir=./logs_modes $args
done





