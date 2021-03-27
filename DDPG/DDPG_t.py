import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
import numpy as np
# from ConditionerEnv import Env
from collections import defaultdict
import pandas as pd 
import os
import json

#hyper parameters
MAX_EPISODES = 20000                #episodes for train
LR_A = 0.003                    #learning rate of actor
LR_C = 0.002                    #learning rate of critic
GAMMA = 0.0                    #reward discount factor
# TAU = 0.1                        #update network parameter with TAU * w + (1-TAU) * w'
MEMORY_CAPACITY = 2000            #max memory capacity
BATCH_SIZE = 1024                    #batch size for sampling from experience pool 
CKPT_DIR = './model/'
SAVE_MODEL_EVERY = 2


#DDPG
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        # graph something
        tf.reset_default_graph()
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 2), dtype=np.float32)
        #for memory update
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.done = tf.placeholder(tf.float32, [None, 1], 'done')

        # self.a for evaluation, a_ for target, build with self._build_a
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            # a_ = self._build_a(self.S_, scope='target', trainable=False)

        # self.q for evaluation, q_ for target, build with self._build_c
        # q_ build with self._c and a_
        with tf.variable_scope('Critic'):
            self.q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            # q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # obtain params of all networks for soft replacement
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        # self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        # self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # self.soft_replace = [tf.assign(t, (1-TAU)*t + TAU*e) for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]
        # self.soft_replace_a = [tf.assign(t, (1-TAU)*t + TAU*e) for t, e in zip(self.at_params, self.ae_params)]
        # self.soft_replace_e = [tf.assign(t, (1-TAU)*t + TAU*e) for t, e in zip(self.ct_params, self.ce_params)]

        # q_target = self.R + GAMMA * q_ * (1.0 - self.done)
        q_target = self.R
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
        # update critic network with TD error
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        # update actor nework with -q
        a_loss = - tf.reduce_mean(self.q)
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        # defination saver, find saved checkpoint, load saved model params
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        if not os.path.exists(CKPT_DIR):
            os.makedirs(CKPT_DIR)
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print('==========restore params==========')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        # 最近五次数据, dataframe
        self.last5 = []
        # 算法是否已有正在执行的动作
        self.is_any_action_executing = False
        # 需要保存的执行动作时的环境状态
        self.temp_s = 0
        # 需要保存的执行动作
        self.temp_a = 0

    def choose_action(self, data):
        # json转df
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
                # 当前状态
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
                #print('状态s:',s)
                # 如果没有正在执行的动作，此时环境稳定
                # 保存当前状态到self.temp_s，保存当前执行动作到self.temp_a，下发动作，更改is_any_action_executing状态
                if not self.is_any_action_executing:
                    action = self.sess.run(self.a, feed_dict={self.S: s[np.newaxis, :]})
                    action = np.squeeze(action)
                    action = np.array([round(i, 1) for i in action])

                    # 3 actions
                    # PID1_action = [df.loc[0, '风机1转速'], df.loc[0, '压缩机1容量'], df['冷凝风机1转速']]
                    # PID2_action = [df.loc[0, '风机2转速'], df.loc[0, '压缩机2容量'], df['冷凝风机2转速']]
                    # PID_action = np.array([(float(PID1_action[i]) + float(PID2_action[i])) / 2 for i in range(3)])

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
                            # 可下发动作，或者不下发动作
                            return np.array([0, 0])
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
                        # self.store_transition(s, action, -100, s, False)
                        # 仅下发心跳信号
                        return np.array([0, 0])

                # 如果有正在执行的动作，此时环境稳定
                # 计算step反馈，保存step经验，self.is_any_action_executing状态False，下发PID控制动作
                else:
                    reward = self.reward_calculation(df, self.suitable_flag)
                    
                    self.store_transition(self.temp_s, self.temp_a, reward, s, 1)
                    self.is_any_action_executing = False
                    #print('环境稳定，有执行动作，计算step反馈，保存经验，下发PID控制动作')
                    print('本轮较好动作的惩罚是：', reward)
                    with open('./reward.txt', 'a') as f:
                        f.write(str(reward) + '\n')
                    # 下发PID控制动作
                    return np.array([0, 0])
            # 环境不稳定
            else:
                # 仅下发心跳信号
                print('环境不稳定，仅下发心跳信号')
                return np.array([0, 0])

    def predict(self, s, a):
        return self.sess.run(self.q, feed_dict={self.S:s[np.newaxis, :], self.a:a})

    def learn(self):
        # self.sess.run(self.soft_replace)
        # self.sess.run(self.soft_replace_a)
        # self.sess.run(self.sort_replace_c)
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        batch_data = self.memory[indices, :]
        bs = batch_data[:, :self.s_dim]
        ba = batch_data[:, self.s_dim:self.s_dim + self.a_dim]
        br = batch_data[:, -self.s_dim - 2: -self.s_dim - 1]
        bs_ = batch_data[:, -self.s_dim - 1: -1]
        bd = batch_data[:, -1]
        bd = np.reshape(bd, (BATCH_SIZE, 1))

        self.sess.run(self.atrain, feed_dict={self.S: bs})
        self.sess.run(self.ctrain, feed_dict={self.S: bs, self.a:ba, self.R:br, self.S_:bs_, self.done:bd})

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, [done]))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1
        # 每十条经验训练一次
        if self.pointer % 10 == 3:
            print('learn')
            self.learn()
            self.saver.save(self.sess, CKPT_DIR + 't.ckpt')

    def reward_calculation(self, df, suitable_flag):
        # 基本reward计算
        kw = float(df.loc[0, '总有功功率'])
        songFengError = abs(float(df.loc[0, '送风平均温度']) - float(df.loc[0, '送风温度设定']))
        huiFengError = abs(float(df.loc[0, '回风平均温度']) - float(df.loc[0, '回风温度设定']))
        # print('kw:{}, songFengError:{}, huiFengError:{}'.format(kw, songFengError, huiFengError))
        reward = - kw * 10 - songFengError * 5 - huiFengError * 5

        #故障reward计算
        if float(df.loc[0, '冷凝器驱动故障']) or float(df.loc[0, '冷凝器压力传感器故障']) or float(df.loc[0, '冷凝器风机故障']) or float(df.loc[0, '冷凝器喷淋高压故障']) or float(df.loc[0, '冷凝器低压差故障']) or float(df.loc[0, '压缩机1运行状态']) != 2 or float(df.loc[0, '压缩机2运行状态']) != 2:
            reward -= 50

        print('本次reward为：',reward)
        return reward
        # if suitable_flag:
        #     return -5
        # else:
        #     return -15

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net1 = tf.layers.dense(s, 16, activation=tf.nn.tanh, name='l1', trainable=trainable)
            # net2 = tf.layers.dense(net1, 8, activation=tf.nn.tanh, name='l2', trainable=trainable)
            netn = tf.layers.dense(net1, 8, activation=tf.nn.tanh, name='ln', trainable=trainable)
            a = tf.layers.dense(netn, self.a_dim, activation=tf.nn.sigmoid,name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            w1_s = tf.get_variable('w1_s', [self.s_dim, 16], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, 16], trainable=trainable)
            b1 = tf.get_variable('b1', [1, 16], trainable=trainable)
            net1 = tf.nn.tanh(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            #net2 = tf.layers.dense(net1, 32, activation=tf.nn.relu, name='l2', trainable=trainable)
            netn = tf.layers.dense(net1, 8, activation=tf.nn.tanh, name='ln', trainable=trainable)
            return tf.multiply(tf.layers.dense(netn, 1, activation=tf.nn.sigmoid, trainable=trainable), -80)

# def DDPG4debug():
#     pass

# def main():
#     DDPG4debug()

# if __name__ == '__main__':
#     main()










