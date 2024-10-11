import numpy as np
from tqdm import tqdm

class HMM(object):
    def __init__(self, tag2idx, idx2tag):
        self.tag2idx = tag2idx
        self.idx2tag = idx2tag

        self.num_states = len(self.tag2idx) # 21
        self.num_char = 65536  # 字符数
        self.eps = 1e-16  # 无穷小量

        # 状态转移矩阵A：A[i][j]表示从i状态转移到j状态的概率
        self.A = np.zeros((self.num_states, self.num_states))  # [21, 21] 状态转移概率矩阵
        # 观测概率矩阵B：B[i][j]表示i状态下生成j字符
        self.B = np.zeros((self.num_states, self.num_char))  # [21, 65536] 观测概率矩阵
        # 初始状态概率pi：pi[i]表示初始时刻为i状态的概率
        self.pi = np.zeros(self.num_states)  # [21,] 初始隐状态概率

    def train(self, train_data):
        print('start Training：') # 即统计数据
        for i in tqdm(range(len(train_data))):
            for j in range(len(train_data[i][0])):
                cur_char = train_data[i][0][j]
                cur_tag = train_data[i][1][j]

                # 统计 B 矩阵：B[i][j]表示i状态下生成j字符
                self.B[self.tag2idx[cur_tag]][ord(cur_char)] += 1

                # 统计 pi 矩阵：pi[i]表示初始时刻为i状态的概率
                if j == 0:
                    # 若是文本段的第一个字符，统计pi矩阵
                    self.pi[self.tag2idx[cur_tag]] += 1
                    continue

                # 统计 A 矩阵：A[i][j]表示从i状态转移到j状态的概率
                pre_tag = train_data[i][1][j - 1]
                self.A[self.tag2idx[pre_tag]][self.tag2idx[cur_tag]] += 1

        # 归一化为概率分布
        self.A = self.A / (np.sum(self.A, axis=1, keepdims=True) + self.eps)
        self.B = self.B / (np.sum(self.B, axis=1, keepdims=True) + self.eps)
        self.pi = self.pi / (np.sum(self.pi) + self.eps)

        print('Training Finished！')

    # 使用维特比算法进行解码
    def viterbi(self, string):
        # 初始化delta矩阵，delta[t][i]表示t时刻状态为i的概率
        delta = np.zeros((len(string), self.num_states))
        # 初始化psi矩阵，psi[t][i]表示t时刻状态为i的最大概率路径的前一个状态
        psi = np.zeros((len(string), self.num_states), dtype=int)

        # 初始化第一个字符的delta值
        delta[0] = self.pi * self.B[:, ord(string[0])]

        # 递推计算delta和psi
        for t in range(1, len(string)):
            for j in range(self.num_states):
                prob = delta[t-1] * self.A[:, j] * self.B[j, ord(string[t])]
                psi[t, j] = np.argmax(prob)
                delta[t, j] = np.max(prob)

        # 回溯找到最优路径
        path = np.zeros(len(string), dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(len(string)-2, -1, -1): 
            path[t] = psi[t+1, path[t+1]]

        # 将路径转换为标签序列
        results = [self.idx2tag[idx] for idx in path]
        return results

    def predict(self, s):
        results = self.viterbi(s)
        for i in range(len(s)):
            print(s[i] + results[i], end=' | ')

    def valid(self, valid_data):
        y_pred = []
        # 遍历验证集每一条数据，使用维特比算法得到预测序列，并加到列表中
        for i in range(len(valid_data)):
            y_pred.append(self.viterbi(valid_data[i][0]))
        return y_pred