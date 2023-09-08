combinations=(
    # "-c --bob-hand" #conditional 2hist hand/wrist
    # "-c" #conditional 2hist alljoints
    # "--bob-hand --one-hist" #marginal 1hist hand/wrist
    "--one-hist" #marginal 1hist alljoints
)

for args in "${combinations[@]}"; do
    python finetune_intent_forecaster.py --log-dir=./logs_ft_intent --batch-size=64 $args
    python finetune_hr.py  --log-dir=./logs_ft_intent_hr $args
done




