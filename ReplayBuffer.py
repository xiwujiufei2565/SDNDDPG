# 导入相关模块
from collections import deque
import random

# 建立经验缓冲池
class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size  # 经验池最大大小
        self.num_experiences = 0    # 当前经验池的中经验的数量
        self.buffer = deque()       # 经验池

    # 获取batsize的历史经验
    def getBatch(self, batch_size):
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences) # 当经验池不足batch_size时,全部返回
        else:
            return random.sample(self.buffer, batch_size)   # 当经验池充足时，随机返回batch_size的经验

    # 返回缓冲池的大小
    def getSize(self):
        return self.buffer_size

    # 给缓冲池添加内容
    def add(self, state, action, reward, new_state):
        experience = (state, action, reward, new_state)
        if self.num_experiences < self.buffer_size:
            # 如果缓冲区没有存满，则直接进行存储
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            # 如果缓冲区存满了，则将之前的第一个删除
            self.buffer.popleft()
            self.buffer.append(experience)

    # 返回当前存储经验的大小
    def getCount(self):
        return self.num_experiences

    # 重置缓冲区
    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0