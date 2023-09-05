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
    python finetune_cond_forecaster.py --log-dir=./logs_new_ft --batch-size=64 $args
done





