combinations=(
    "-c --bob-hand"
    "-c"
    "--bob-hand"
    ""
)

for args in "${combinations[@]}"; do
    python finetune_cond_forecaster.py --log-dir=./logs_ft_20 --batch-size=64 $args
done





