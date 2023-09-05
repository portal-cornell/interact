combinations=(
    "-c --bob-hand"
    "-c"
    "--bob-hand"
    ""
)
for args in "${combinations[@]}"; do
    python finetune_cond_trans.py --log-dir=./logs_trans_ft --batch-size=64 $args
done





