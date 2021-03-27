import time
import os
import numpy as np 
import json
import pandas as pd 
from collections import defaultdict
from DDPG_t import DDPG

data_root = './data_0315/'
data_list = os.listdir(data_root)

ddpg = DDPG(2, 7, 100)

flag = 1

for j in range(100):
    for i in data_list:
        with open(data_root + i, 'r') as f:
            data = f.read()
        print('-----{}-----'.format(flag))
        flag += 1
        a = ddpg.choose_action(data)
        #print(a)
