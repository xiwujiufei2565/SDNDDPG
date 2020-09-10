# 导入相关模块
import numpy as np
import math
import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, Add
from keras.models import Model
from keras.optimizers import Adam

# 常量定义
HIDDEN1_UNITS = 50 # 第一层神经元
HIDDEN2_UNITS = 50 # 第二层神经元

# Critic神经网络
class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size, tau, learning_rate):
        self.sess = sess
        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate
        self.action_size = action_size

        K.set_session(self.sess)

        # 创建神经网络
        self.model, self.state, self.action = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_state, self.target_action = self.create_critic_network(state_size, action_size)
        # 计算梯度
        self.action_grads = tf.gradients(self.model.output, self.action)
        self.sess.run(tf.initialize_all_variables())

    # 训练eval网络
    def tarin(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]   # 返回的是dq/da的梯度

    # 目标网络参数的更新
    def target_train(self):
        eval_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(eval_weights)):
            target_weights[i] = self.tau * eval_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    # 创建网络结构
    def create_critic_network(self, state_szie, action_size):
        # debug
        print("Now we will creat Critic network")
        S = Input(shape=[state_szie])
        A = Input(shape=[action_size])
        s1 = Dense(HIDDEN1_UNITS, activation='tanh')(S)
        # 合并之前使用一个线性激活函数，保存其特征
        s2 = Dense(HIDDEN2_UNITS, activation='tanh')(s1)
        a1 = Dense(HIDDEN2_UNITS, activation='tanh')(A)
        # 将两个输出层进行合并，即简单的将对应位置相加
        h1 = keras.layers.add([s2, a1])
        h2 = Dense(HIDDEN2_UNITS, activation='relu')(h1)
        V = Dense(1, activation='relu')(h2)
        model = Model(input=[S, A], output=V)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model, S, A



