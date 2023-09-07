combinations=(
    # "-c --bob-hand" #conditional 2hist hand/wrist
    # "-c" #conditional 2hist alljoints
    "--bob-hand --one-hist" #marginal 1hist hand/wrist
    "--one-hist" #marginal 1hist alljoints
)
for args in "${combinations[@]}"; do
    python train_intent_forecaster.py --log-dir=./logs_intent_pretrain $args
done





