@echo off
D:
cd D:\yifei\Documents\Codes_on_GitHub\crime_models
for %%I in (0,1,2,3,4,5,6) do python train_eval_models.py --model_name ARMA --hyper_parameters 7 %%I
pause
