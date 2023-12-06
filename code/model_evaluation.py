import lightgbm as lgb 
from config import *

paths = Paths()
model = lgb.Booster(model_file='Baseline.txt')
print("test")
