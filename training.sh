python train_mrt_charm.py --one-hist --log-dir=./logs_amass
python train_cond_forecaster.py --one-hist -c --no-amass --bob-hand --log-dir=./logs_tmp
python train_mrt_charm.py --log-dir=./logs_amass
python train_mrt_charm.py -c --log-dir=./logs_amass