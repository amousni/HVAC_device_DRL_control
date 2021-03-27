# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:30:48 2021

@author: al
"""

"""
Proximal Policy Optimization (PPO)
----------------------------
A simple version of Proximal Policy Optimization (PPO) using single thread.
PPO is a family of first-order methods that use a few other tricks to keep new policies close to old.
PPO methods are significantly simpler to implement, and empirically seem to perform at least as well as TRPO.
Reference
---------
Proximal Policy Optimization Algorithms, Schulman et al. 2017
High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016
Emergence of Locomotion Behaviours in Rich Environments, Heess et al. 2017
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials
Environment
-----------
Openai Gym Pendulum-v0, continual action space
Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0
To run
------
python tutorial_PPO.py --train/test
"""
import argparse
import os
import time

#import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Sequential, layers
from tensorflow import keras

from collections import defaultdict
import json
import pandas as pd

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_ID = 'Pendulum-v0'  # environment id
RANDOM_SEED = 1  # random seed
RENDER = False  # render while training

ALG_NAME = 'PPO'
TRAIN_EPISODES = 1000  # total number of episodes for training
TEST_EPISODES = 10  # total number of episodes for testing
MAX_STEPS = 200  # total number of steps for each episode
GAMMA = 0.9  # reward discount
LR_A = 0.0001  # learning rate for actor
LR_C = 0.0002  # learning rate for critic
BATCH_SIZE = 32  # update batch size
ACTOR_UPDATE_STEPS = 10  # actor update steps
CRITIC_UPDATE_STEPS = 10  # critic update steps

# ppo-penalty parameters
KL_TARGET = 0.01
LAM = 0.5

# ppo-clip parameters
EPSILON = 0.2


###############################  PPO  ####################################


class PPO(object):
    """
    PPO class
    """
    def __init__(self, action_dim, state_dim, action_bound, method='clip'):
        # critic
        with tf.name_scope('critic'):
            self.critic = Sequential([
                layers.Dense(64, tf.nn.relu, name='layer1', input_dim=state_dim),
                layers.Dense(64, tf.nn.relu, name='layer2'),
                layers.Dense(1, tf.nn.sigmoid, name='outlayer'),
                layers.Lambda(lambda x: x * (-80), name='lambda')
                ])


        # actor
        with tf.name_scope('actor'):
            self.actor = Sequential([
                layers.Dense(64, tf.nn.relu, name='layer1', input_dim=state_dim),
                layers.Dense(64, tf.nn.relu, name='layer2'),
                layers.Dense(action_dim, tf.nn.sigmoid, name='outlayer'),
                layers.Lambda(lambda x: x * action_bound, name='lambda')
                ])
            logstd = tf.Variable(np.zeros(action_dim, dtype=np.float32))
        self.actor.weights.append(logstd)
        self.actor.logstd = logstd

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

        self.method = method
        if method == 'penalty':
            self.kl_target = KL_TARGET
            self.lam = LAM
        elif method == 'clip':
            self.epsilon = EPSILON

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []
        self.action_bound = action_bound
        
        # 最近五次数据, dataframe
        self.last5 = []
        # 算法是否已有正在执行的动作
        self.is_any_action_executing = False
        # 需要保存的执行动作时的环境状态
        self.temp_s = 0
        # 需要保存的执行动作
        self.temp_a = 0
        # 更新频率
        self.pointer = 0
        
        if 'model' in os.listdir('./'):
            self.load()
            print('==============load params=============')

    def train_actor(self, state, action, adv, old_pi):
        """
        Update policy network
        :param state: state batch
        :param action: action batch
        :param adv: advantage batch
        :param old_pi: old pi distribution
        :return: kl_mean or None
        """
        with tf.GradientTape() as tape:
            mean, std = self.actor(state), tf.exp(self.actor.logstd)
            pi = tfp.distributions.Normal(mean, std)

            ratio = tf.exp(pi.log_prob(action) - old_pi.log_prob(action))
            surr = ratio * adv
            if self.method == 'penalty':  # ppo penalty
                kl = tfp.distributions.kl_divergence(old_pi, pi)
                kl_mean = tf.reduce_mean(kl)
                loss = -(tf.reduce_mean(surr - self.lam * kl))
            else:  # ppo clip
                loss = -tf.reduce_mean(
                    tf.minimum(surr,
                               tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv)
                )
        a_gard = tape.gradient(loss, self.actor.weights)
        self.actor_opt.apply_gradients(zip(a_gard, self.actor.weights))

        if self.method == 'kl_pen':
            return kl_mean

    def train_critic(self, reward, state):
        """
        Update actor network
        :param reward: cumulative reward batch
        :param state: state batch
        :return: None
        """
        reward = np.array(reward, dtype=np.float32)
        with tf.GradientTape() as tape:
            advantage = reward - self.critic(state)
            loss = tf.reduce_mean(tf.square(advantage))
        grad = tape.gradient(loss, self.critic.weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.weights))

    def update(self):
        """
        Update parameter with the constraint of KL divergent
        :return: None
        """
        s = np.array(self.state_buffer, np.float32)
        a = np.array(self.action_buffer, np.float32)
        r = np.array(self.cumulative_reward_buffer, np.float32)
        mean, std = self.actor(s), tf.exp(self.actor.logstd)
        pi = tfp.distributions.Normal(mean, std)
        adv = r - self.critic(s)

        # update actor
        if self.method == 'kl_pen':
            for _ in range(ACTOR_UPDATE_STEPS):
                kl = self.train_actor(s, a, adv, pi)
            if kl < self.kl_target / 1.5:
                self.lam /= 2
            elif kl > self.kl_target * 1.5:
                self.lam *= 2
        else:
            for _ in range(ACTOR_UPDATE_STEPS):
                self.train_actor(s, a, adv, pi)

        # update critic
        for _ in range(CRITIC_UPDATE_STEPS):
            self.train_critic(r, s)

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.cumulative_reward_buffer.clear()
        self.reward_buffer.clear()

    def choose_action(self, data, greedy=False):
        """
        Choose action
        :param state: state
        :param greedy: choose action greedy or not
        :return: clipped action
        """
        
        c = defaultdict(list)
        data = json.loads(data)['params']
        for j in data:
            c[j['pointName']].append(j['originalValue'])
        c = dict(c)
        df = pd.DataFrame.from_dict(c)
        
        # 历史数据不足5条，不进行动作下发
        if len(self.last5) < 5:
            print('recent data less than 5')
            self.last5.append(df)
            # 仅下发心跳信号
            return np.array([0, 0])
        else:
            # 删除老数据，保存新数据
            self.last5.pop(0)
            self.last5.append(df)
            # 近5分钟内的回风温度，判断是否环境稳定
            last5_songFengTemperature_list = [float(i.loc[0, '回风平均温度']) for i in self.last5]
            # 环境稳定
            if max(last5_songFengTemperature_list) - min(last5_songFengTemperature_list) <= 0.6:
                try:
                    test_temperature = df.loc[0, '环境温度']
                except:
                    print('环境温度缺失')
                    return np.array([0, 0])
                
                s = [df.loc[0, '送风平均温度'], df.loc[0, '回风平均温度'], df.loc[0, '送风平均湿度'], df.loc[0, '回风平均湿度'], df.loc[0,'A相输出有功功率'],
                     df.loc[0, 'B相输出有功功率'], df.loc[0,'C相输出有功功率'], df.loc[0, '环境温度'], df.loc[0, '环境湿度']]
                s = [float(i) for i in s]
                # 将三个有功功率做平均
                s = np.array([s[0] - 14, (s[1] - 22) / 2, (s[2] - 68) / 10, (s[3] - 42) / 2, (s[4] + s[5] + s[6]) / 3, (s[7] - 10) / 30, s[8] / 100])
        
                state = s[np.newaxis, :].astype(np.float32)
                # 如果没有正在执行的动作，此时环境稳定
                # 保存当前状态到self.temp_s，保存当前执行动作到self.temp_a，下发动作，更改is_any_action_executing状态
                if not self.is_any_action_executing:
                    mean, std = self.actor(state), tf.exp(self.actor.logstd)
                    if greedy:
                        action = mean[0]
                    else:
                        pi = tfp.distributions.Normal(mean, std)
                        action = tf.squeeze(pi.sample(1), axis=0)[0]  # choosing action
                    action = np.clip(action, -self.action_bound, self.action_bound)
                    action = np.array([round(i, 1) for i in action])
                    
                    # 2 actions
                    PID1_action = [df.loc[0, '压缩机1容量'], df['冷凝风机1转速']]
                    PID2_action = [df.loc[0, '压缩机2容量'], df['冷凝风机2转速']]
                    PID_action = np.array([(float(PID1_action[i]) + float(PID2_action[i])) / 2 for i in range(2)])

                    print('PID动作：',PID_action)
                    print('我们发出的动作：',action)

                    # 判断当前动作是否与PID动作相差不大
                    if abs(PID_action[0] - action[0]) <= 15 and abs(PID_action[1] - action[1]) <= 15:
                        if abs(PID_action[0] - action[0]) <= 3 and abs(PID_action[1] - action[1]) <= 5:
                            self.temp_s = s
                            self.temp_a = action
                            self.is_any_action_executing = True
                            #print('环境稳定，动作稳定，没有执行动作，下发动作')
                            # 如果压缩机动作与冷凝风机转速一增一减，则视为较好动作
                            if (PID_action[0] - action[0]) * (PID_action[1] - action[1]) < 0:
                                self.suitable_flag = 1
                            else:
                                self.suitable_flag = 0
                            return action
                        else:
                            #print('环境稳定，动作一般，不下发动作')
                            action_1_error = abs(PID_action[1] - action[1])
                            action_0_error = abs(PID_action[0] - action[0])
                            # action_2_error = abs(PID_action[2] - action[2])
                            reward = -(action_0_error + action_1_error) * 0.5 - 20
                            print('本轮一般动作的惩罚是：',reward)
                            with open('./reward.txt', 'a') as f:
                                f.write(str(reward) + '\n')
                            self.store_transition(s, action, reward, s, 1)
                    # action和PID差距有点大，返回心跳信号，不下发动作，但可以将反馈设置为很大，保存step数据
                    else:
                        #print('环境稳定，动作不稳定，不执行动作，保存负反馈数据')
                        action_1_error = abs(PID_action[1] - action[1])
                        action_0_error = abs(PID_action[0] - action[0])
                        # action_2_error = abs(PID_action[2] - action[2])
                        reward = -(action_0_error + action_1_error) * 0.5 - 20
                        print('本轮不合格动作的惩罚是：',reward)
                        with open('./reward.txt', 'a') as f:
                            f.write(str(reward) + '\n')
                        self.store_transition(s, action, reward, s, 1)
                        return np.array([0, 0])

                # 如果有正在执行的动作，此时环境稳定
                # 计算step反馈，保存step经验，self.is_any_action_executing状态False，下发PID控制动作
                else:
                    reward = self.reward_calculation(df, self.suitable_flag)
                    
                    self.store_transition(self.temp_s, self.temp_a, reward, s, 1)
                    self.is_any_action_executing = False
                    print('本轮较好动作的惩罚是：', reward)
                    with open('./reward.txt', 'a') as f:
                        f.write(str(reward) + '\n')
                    return np.array([0, 0])
            # 环境不稳定
            else:
                # 仅下发心跳信号
                print('环境不稳定，仅下发心跳信号')
                return np.array([0, 0])




    def reward_calculation(self, df, suitable_flag):
        # kw = float(df.loc[0, '总有功功率'])
        # songFengError = abs(float(df.loc[0, '送风平均温度']) - float(df.loc[0, '送风温度设定']))
        # huiFengError = abs(float(df.loc[0, '回风平均温度']) - float(df.loc[0, '回风温度设定']))
        # print('kw:{}, songFengError:{}, huiFengError:{}'.format(kw, songFengError, huiFengError))
        # reward = kw * 100 - songFengError * 10 - huiFengError * 10
        # print('本次reward为：',reward)
        # return reward
        if suitable_flag:
            return -5
        else:
            return -15                        
                        
                        

    def save(self):
        """
        save trained weights
        :return: None
        """
        path = os.path.join('model', 'A64-64_C64-64')
        if not os.path.exists(path):
            os.makedirs(path)
        self.actor.save_weights(os.path.join(path, 'actor.h5'))
        self.critic.save_weights(os.path.join(path, 'critic.h5'))
        # tl.files.save_weights_to_hdf5(os.path.join(path, 'actor.hdf5'), self.actor)
        # tl.files.save_weights_to_hdf5(os.path.join(path, 'critic.hdf5'), self.critic)

    def load(self):
        """
        load trained weights
        :return: None
        """
        path = os.path.join('model', 'A64-64_C64-64')
        self.actor.load_weights(os.path.join(path, 'actor.h5'))
        self.critic.load_weights(os.path.join(path, 'critic.h5'))
        # tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor.hdf5'), self.actor)
        # tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic.hdf5'), self.critic)

    def store_transition(self, state, action, reward, s, done):
        """
        Store state, action, reward at each step
        :param state:
        :param action:
        :param reward:
        :return: None
        """
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.finish_path(s, done)
        self.pointer += 1
        if self.pointer % 10 == 3:
            self.update()
            self.save()

    def finish_path(self, next_state, done):
        """
        Calculate cumulative reward
        :param next_state:
        :return: None
        """
        if done:
            v_s_ = 0
        else:
            v_s_ = self.critic(np.array([next_state], np.float32))[0, 0]
        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        self.cumulative_reward_buffer.extend(discounted_r)
        self.reward_buffer.clear()

    
    
    
    
    

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            