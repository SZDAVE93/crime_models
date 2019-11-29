@echo off
D:
cd D:\yifei\Documents\Codes_on_GitHub\crime_models
for %%I in (910) do python train_eval_models.py --model_name LSTM --hyper_parameters 128 3 --iters 10000 --learning_rate 1e-3 --train_end %%I
pause
