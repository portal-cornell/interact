combinations=(
    # "-c --bob-hand" #conditional 2hist hand/wrist
    # "-c" #conditional 2hist alljoints
    # "--bob-hand --one-hist" #marginal 1hist hand/wrist
    # "--one-hist" #marginal 1hist alljoints
    # "--bob-hand" #marginal 2hist hand/wrist
    # "" #marginal 2hist alljoints

    "--bob-hand --one-hist" #marginal 1hist hand/wrist
    "--one-hist" #marginal 1hist alljoints

    # "-c --bob-hand" #conditional 2hist hand/wrist
    # "-c" #conditional 2hist alljoints
    # "-c --bob-hand --align_rep" #conditional 2hist hand/wrist
    # "-c --align_rep" #conditional 2hist alljoints
)

for args in "${combinations[@]}"; do
    # python finetune_intent_forecaster.py --log-dir=./logs_ft_intent_withCMU $args
    python finetune_hr.py  --log-dir=./logs_ft_intent_hr_align_orient $args
done





