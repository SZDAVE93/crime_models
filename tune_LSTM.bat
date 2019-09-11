@echo off
D:
cd D:\yifei\Documents\Codes_on_GitHub\crime_models
for %%I in (2,4,8,16,32,64,128,256) do python train_eval_models.py --model_name LSTM --hyper_parameters %%I 1 --iters 10000 --learning_rate 1e-3 --eval_days 21
for %%I in (2,4,8,16,32,64,128,256) do python train_eval_models.py --model_name LSTM --hyper_parameters %%I 2 --iters 10000 --learning_rate 1e-3 --eval_days 21
for %%I in (2,4,8,16,32,64,128,256) do python train_eval_models.py --model_name LSTM --hyper_parameters %%I 4 --iters 10000 --learning_rate 1e-3 --eval_days 21
for %%I in (2,4,8,16,32,64,128,256) do python train_eval_models.py --model_name LSTM --hyper_parameters %%I 8 --iters 10000 --learning_rate 1e-3 --eval_days 21
pause
