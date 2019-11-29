@echo off
python train_eval_models.py --model_name TCP --hyper_parameters 7 1 3 --learning_rate 1e-6 --iters 10000
pause
