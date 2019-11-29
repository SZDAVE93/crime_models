@echo off
D:
cd D:\yifei\Documents\Codes_on_GitHub\crime_models
for %%I in (910) do python train_eval_models.py --model_name Linear --train_end %%I --learning_rate 1e-4 --iters 1000
pause
