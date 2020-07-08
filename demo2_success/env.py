import numpy as np


class Env:
    def __init__(self, run_time=300):
        self.run_time = run_time  # 模拟运行总时间（单位分钟）
        self.current_time = -1  # 当前时刻（单位分钟）
        self.done = False  # 结束标志（current_time>=run_time）
        self.flag1 = True  # 出租车1还未到达上车点
        self.flag2 = True  # 出租车1还未到达上车点
        self.connect = np.loadtxt('./connect16.csv', delimiter=',', dtype=int)  # 环境两个地点的连通表
        self.taxi1_loc = 0  # 出租车1的位置
        self.taxi2_loc = 15  # 出租车2的位置
        self.reward1 = 0  # 出租车1的奖励
        self.reward2 = 0  # 出租车2的奖励
        self.observation = np.array([[0, 0], [0, 0]])  # 环境状态

    def reset(self, run_time=300):
        self.run_time = run_time
        self.current_time = -1
        self.done = False
        self.flag1 = True
        self.flag2 = True
        self.connect = np.loadtxt('./connect16.csv', delimiter=',', dtype=int)
        self.taxi1_loc = 0
        self.taxi2_loc = 15
        self.reward1 = 0
        self.reward2 = 0
        self.observation = np.array([[0, 0], [0, 0]])  # 环境状态

    def refresh_env(self):
        # 时间流逝一分钟
        self.current_time += 1
        if self.current_time >= self.run_time:
            self.done = True
        self.get_observation()
        return self.observation

    # 执行决策
    def taxi_refresh(self, action):
        self.reward1 = 0.0
        self.reward2 = 0.0

        # 更新出租车1的位置
        next_loc1 = self.connect[self.taxi1_loc, action[0]]
        if next_loc1 != -1:
            self.taxi1_loc = next_loc1
        # 是否到达上车点
        if self.taxi1_loc == 11 and self.flag1:
            self.flag1 = False
            self.reward1 = 3.0
        # 是否到达下车点
        if self.taxi1_loc == 7 and not self.flag1:
            self.done = True
            self.reward1 = 4.0

        # 更新出租车2的位置
        next_loc2 = self.connect[self.taxi2_loc, action[1]]
        if next_loc2 != -1:
            self.taxi2_loc = next_loc2
        # 是否到达上车点
        if self.taxi2_loc == 11 and self.flag2:
            self.flag2 = False
            self.reward2 = 3.0
        # 是否到达下车点
        if self.taxi2_loc == 7 and not self.flag2:
            self.done = True
            self.reward2 = 4.0

        self.get_observation()
        return self.observation, [self.reward1, self.reward2]

    # 生成 observation
    def get_observation(self):
        if self.flag1:
            self.observation = np.array([[(float(self.taxi1_loc)) / 16, 0],
                                         [(float(self.taxi2_loc)) / 16, 0]])
        else:
            self.observation = np.array([[(float(self.taxi1_loc)) / 16, 1],
                                         [(float(self.taxi2_loc)) / 16, 1]])
