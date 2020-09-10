# 导入相关模块
import numpy as np
import tensorflow as tf
import json
import warnings
warnings.filterwarnings("ignore")
from SDNDDPG import Actor
from SDNDDPG import Critic
from SDNDDPG import ReplayBuffer
from SDNDDPG import Sote
from SDNDDPG import OUNoise
from matplotlib import pyplot as plt

# 将网络的输出结果action，转换为0/1序列
def transform_form(output, num):
    # output---网络的输出（0-1的概率）    num---多少节点要被转换
    output = np.array(output)
    output = output.ravel()
    index = []
    for i in range(num):
        max_index = 0
        max = -999999
        for j in range(len(output)):
            if j not in index and output[j] > max:
                max_index = j
                max = output[j]
        index.append(max_index)
    for i in range(len(output)):
        if i not in index:
            output[i] = 0
        else:
            output[i] = 1
    return output

# 归一化
def Normalize(data):
    # data---需要归一化的数据集
    mx = data.max(axis=1).reshape(-1, 1)
    mn = data.min(axis=1).reshape(-1, 1)
    data = (data - mn) / (mx - mn)
    return data

# 强化学习主函数
def play():
    # 相关常量的定义
    BUFFER_SIZE = 100000  # 缓冲池的大小
    BATCH_SIZE = 32  # batch_size的大小
    GAMMA = 0.99  # 折扣系数
    TAU = 0.001  # target网络软更新的速度
    LR_A = 0.001  # Actor网络的学习率
    LR_C = 0.001  # Critic网络的学习率

    # 相关变量的定义
    vertex_num = 12  # 顶点的个数
    action_dim = vertex_num  # 动作的维度---SDN的部署序列
    state_dim = vertex_num * vertex_num  # 状态的维度---流量在链路上的分配信息
    reward = 0  # 奖励值
    episode = 10000  # 迭代的次数
    step = 1000  # 每次需要与环境交互的步数
    total_step = 0  # 总共运行了多少步
    SDN_ratio = 0.4 # SDN节点部署的比率
    SDN_num = int(vertex_num * SDN_ratio)  # 需要的SDN节点的数量

    # 可视化集合定义
    reward_list = []    # 记录所有的rewards进行可视化展示
    loss_list = []  # 记录损失函数进行可视化展示
    step_list = []  # 记录每一步的结果

    # 神经网络相关操作定义
    sess = tf.Session()
    from keras import backend as K
    K.set_session(sess)
    OU = OUNoise.OU()   # 引入噪声

    # 初始化四个个网络
    actor = Actor.ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LR_A)
    critic = Critic.CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LR_C)

    # 创建缓冲区
    buff = ReplayBuffer.ReplayBuffer(BUFFER_SIZE)

    # 加载训练数据
    print("Now we load the weight")
    try:
        actor.model.load_weights("src/actormodel.h5")
        critic.model.load_weights("src/criticmodel.h5")
        actor.target_model.load_weights("src/actormodel.h5")
        critic.target_model.load_weights("src/criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    # 开始迭代
    print("Experiment Start.")
    for i in range(episode):
        # 输出当前信息
        print("Episode : " + str(i) + " Replay Buffer " + str(buff.getCount()))
        # 获取初始state
        s_1, u_1 = Sote.get_state_init(vertex_num)
        s_1 = s_1.reshape(1, -1)  # 平铺
        s_t = s_1

        total_reward = 0
        total_loss = 0
        # 开始执行step步
        for t in range(step):
            loss = 0
            # eval网络生成action
            s_t = Normalize(s_t)  # 归一化
            a_t_original = actor.model.predict(s_t)
            a_t_original = Normalize(a_t_original)  # 归一化

            # 添加噪声和探索
            explore = 2000
            if i <= explore:
                a_t_original = OU.function(a_t_original, 1.0, (i / explore), 1.0 - (i / explore))
            else:
                a_t_original = OU.function(a_t_original, 1.0, 0.8, 0.2)

            # 将网络的输出结果转化成为只含0，1的SDN序列
            a_t = transform_form(a_t_original, SDN_num)
            # 环境交互,获取下一个状态---当前流量分配矩阵
            s_t1, u_t1 = Sote.get_state(a_t)
            # 计算在当前部署序列下的奖励值
            r_t = Sote.get_reward(u_1, u_t1)
            # 将该状态转移存储到缓冲池中
            buff.add(s_t, a_t_original, r_t, s_t1)

            # 选取batch_size个样本
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            y_t = np.asarray([e[2] for e in batch])

            # 目标网络的预测q值---相当于y_label
            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            # 训练网络
            states = states.reshape(len(states), -1)
            actions = actions.reshape(len(actions), -1)
            y_t = y_t.reshape(len(y_t), -1)
            loss += critic.model.train_on_batch([states, actions], y_t)  # 计算当前target网络和eval网络的损失值
            a_for_grad = actor.model.predict(states)  # 当前状态下eval网络产生的动作
            grads = critic.tarin(states, a_for_grad)  # 产生的梯度
            actor.train(states, grads)  # 更新eval网络
            actor.target_train()  # 更新target网络
            critic.target_train()  # 更新target网络

            total_reward += r_t
            total_loss += loss
            s_t = s_t1.reshape(1, -1)  # 转移到下一个状态

            # print("Episode", i, "Step", t, "Action", a_t, "Reward", r_t, "Loss", loss)

            total_step += 1

        # 绘图数据添加
        reward_list.append(total_reward)
        step_list.append(i)
        loss_list.append(total_loss/step)

        # 每隔100次保存一次参数
        print("Now we save model")
        actor.model.save_weights("src/actormodel.h5", overwrite=True)
        with open("src/actormodel.json", "w") as outfile:
            json.dump(actor.model.to_json(), outfile)

        critic.model.save_weights("src/criticmodel.h5", overwrite=True)
        with open("src/criticmodel.json", "w") as outfile:
            json.dump(critic.model.to_json(), outfile)

        # 打印相关信息
        print("")
        print("-" * 50)
        print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
        print("TOTAL LOSS @ " + str(i) + "-th Episode  : LOSS " + str(total_loss/step))
        print("Total Step: " + str(total_step))
        print("-" * 50)
        print("")

        # 绘制图像，并保存
        if i != 0 and i % 100 == 0:
            plt.cla()
            plt.plot(step_list, reward_list)
            plt.xlabel("step")
            plt.ylabel("reward")
            plt.title("reward-step")
            img_name = "img/reward/" + str(i) + "-th Episode"
            plt.savefig(img_name)
        if i != 0 and i % 100 == 0:
            plt.cla()   # 清除
            plt.plot(step_list, loss_list)
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title("loss-step")
            img_name = "img/loss/" + str(i) + "-th Episode"
            plt.savefig(img_name)

    # 训练完成之后最后保存信息
    print("Now we save model")
    actor.model.save_weights("src/actormodel.h5", overwrite=True)
    with open("src/actormodel.json", "w") as outfile:
        json.dump(actor.model.to_json(), outfile)

    critic.model.save_weights("src/criticmodel.h5", overwrite=True)
    with open("src/criticmodel.json", "w") as outfile:
        json.dump(critic.model.to_json(), outfile)

    print("Finish.")

if __name__ == '__main__':
    play()
