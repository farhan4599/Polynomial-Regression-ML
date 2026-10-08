# 

Polynomial regression for two problems (var1: turbine power, var2: thermal reservoir).

Mohammed Farhan-BT2024140

## Run
```
pip install -r requirements.txt
python train.py
```

## Method
| Problem | Model | Degree | Selection |
|---|---|---|---|
| var1 | Lasso (alpha=0.003) | 5 | importance-weighted repeated 5-fold CV (test inputs are shifted toward +-1) |
| var2 | Ridge (alpha=0.1) | 12 | repeated 5-fold CV |
