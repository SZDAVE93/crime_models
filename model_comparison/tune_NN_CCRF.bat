@echo off
D:
cd D:\yifei\Documents\Codes_on_GitHub\crime_models
for %%I in (910) do python train_eval_models.py --model_name NN-CCRF --hyper_parameters 128 2 --iters 10000 --learning_rate 1e-4 --kernel_names dis --simi_len 90 --train_end %%I
pause
