@echo off
python train_eval_models.py --model_name NN-CCRF --hyper_parameters 128 2 --iters 10000 --learning_rate 1e-4
pause
