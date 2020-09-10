import random
import numpy as np

# 引入噪声
class OU(object):

    def function(self, x, mu, theta, sigma):
        for i in range(x.shape[1]):
            rand_num = np.random.rand()
            x[0][i] = theta * (mu - x[0][i]) + sigma * rand_num
        return x