@echo off
python train_eval_models.py --model_name LSTM --hyper_parameters 128 1 --iters 10000 --learning_rate 1e-3
pause
