combinations=(
    # "-c --bob-hand" #conditional 2hist hand/wrist
    # "-c" #conditional 2hist alljoints
    # "--bob-hand --one-hist" #marginal 1hist hand/wrist
    # "--one-hist" #marginal 1hist alljoints
    # "--bob-hand" #marginal 2hist hand/wrist
    # "" #marginal 2hist alljoints

    # "--bob-hand --one-hist" #marginal 1hist hand/wrist
    # "--one-hist" #marginal 1hist alljoints - Marginal FT
    # "" 

    # "-c --bob-hand" #conditional 2hist hand/wrist - Conditional FT wrist/hand
    # "-c" #conditional 2hist alljoints - Conditional FT all J
    # "-c --bob-hand --align_rep --align_weight=0.2" #conditional 2hist hand/wrist - Conditional FT wrist/hand align
    "-c --align_rep --align_weight=0.2" #conditional 2hist alljoints
)

for args in "${combinations[@]}"; do
    # python finetune_intent_forecaster.py --log-dir=./logs_ft_intent_noCMU $args
    python finetune_hr.py  --log-dir=./logs_ft_intent_hr_align_test_final $args
done





