@echo off
D:
cd D:\yifei\Documents\Codes_on_GitHub\crime_models
for %%I in (400,580,760,940) do python train_eval_models.py --model_name TCP --hyper_parameters 7 1 3 --learning_rate 1e-5 --iters 10000 --eval_days 1 --train_end %%I
for %%I in (400,580,760,940) do python train_eval_models.py --model_name TCP --hyper_parameters 7 1 3 --learning_rate 1e-5 --iters 10000 --eval_days 7 --train_end %%I
for %%I in (400,580,760,940) do python train_eval_models.py --model_name TCP --hyper_parameters 7 1 3 --learning_rate 1e-5 --iters 10000 --eval_days 14 --train_end %%I
for %%I in (400,580,760,940) do python train_eval_models.py --model_name TCP --hyper_parameters 7 1 3 --learning_rate 1e-5 --iters 10000 --eval_days 21 --train_end %%I
pause
