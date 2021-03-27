import time
import os
import numpy as np 
import json
import pandas as pd 
from collections import defaultdict
from PPO_t import PPO

data_root = '../../DDPG/data_0315/'
data_list = os.listdir(data_root)

ppo = PPO(2, 7, 100)

flag = 1

for j in range(100):
    for i in data_list:
        with open(data_root + i, 'r', encoding='utf-8') as f:
            data = f.read()
        print('-----{}-----'.format(flag))
        flag += 1
        a = ppo.choose_action(data)
        # print(a)