@echo off
D:
cd D:\yifei\Documents\Codes_on_GitHub\crime_models
for %%I in (400,430,460,490,520,550,580,610,640,670,700,730,760,790,820,850,880,910,940,970) do python train_eval_models.py --model_name TCP --hyper_parameters 7 1 3 --learning_rate 1e-6 --iters 10000 --train_end %%I --kernel_names mobi
pause
