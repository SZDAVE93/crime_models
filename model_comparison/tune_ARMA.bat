@echo off
D:
cd D:\yifei\Documents\Codes_on_GitHub\crime_models
for %%I in (400,430,460,490,520,550,580,610,640,670,700,730,760,790,820,850,880,910,940,970) do python train_eval_models.py --model_name ARMA --hyper_parameters 7 0 --train_end %%I
pause
