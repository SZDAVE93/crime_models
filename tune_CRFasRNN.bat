@echo off
D:
cd D:\yifei\Documents\Codes_on_GitHub\crime_models
for %%I in (1,2,4) do python train_eval_models.py --model_name CRFasRNN --hyper_parameters %%I --iters 10000 --learning_rate 1e-8 --kernel_names dis --eval_days 14 --simi_len 90 --train_end 760
for %%I in (1,2,4) do python train_eval_models.py --model_name CRFasRNN --hyper_parameters %%I --iters 10000 --learning_rate 1e-8 --kernel_names dis --eval_days 7 --simi_len 90 --train_end 760
pause
