@echo off
python train_eval_models.py --model_name CRFasRNN --hyper_parameters 6 --iters 10000 --learning_rate 1e-8 --kernel_names poi
