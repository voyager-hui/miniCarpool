import numpy as np


class Env:
    def __init__(self, run_time=300):
        self.done = False  # 结束标志（current_time>=run_time）
        self.run_time = run_time  # 模拟运行总时间（单位分钟）
        self.current_time = -1  # 当前时刻（单位分钟）
        self.observation = np.array([0, 0])  # 当前环境的
        self.taxi_loc = 0  # taxi的位置坐标
        self.reward = 0
        self.flag = True
        self.connect = np.loadtxt('./connect16.csv', delimiter=',', dtype=int)
        self.up_loc = 4
        self.down_loc = 11

    def reset(self, run_time=300):
        self.done = False  # 结束标志（current_time>=run_time）
        self.run_time = run_time  # 模拟运行总时间（单位分钟）
        self.current_time = -1  # 当前时刻（单位分钟）
        self.observation = np.array([0, 0])  # 当前环境的状态
        self.taxi_loc = 0  # taxi的位置坐标
        self.reward = 0
        self.flag = True
        self.connect = np.loadtxt('./connect16.csv', delimiter=',', dtype=int)

    def refresh_env(self):
        # 时间流逝一分钟
        self.current_time += 1
        if self.current_time >= self.run_time:
            self.done = True
        self.get_observation()
        return self.observation

    def new_request(self):
        self.up_loc = np.random.randint(0, 16)
        self.down_loc = np.random.randint(0, 16)

    def taxi_refresh(self, action):
        self.reward = 0.0
        # 更新出租车位置
        next_loc = self.connect[self.taxi_loc, action]
        if next_loc != -1:
            self.taxi_loc = next_loc
        else:
            self.reward = -0.1
        # 是否到达上车点
        if self.taxi_loc == self.up_loc and self.flag:
            self.get_observation()
            self.reward = 4.0
            self.flag = False
            return self.observation, self.reward
        # 是否到达下车点
        if self.taxi_loc == self.down_loc and not self.flag:
            self.get_observation()
            self.reward = 4.0
            self.done = True
            self.new_request()
            return self.observation, self.reward
        self.get_observation()
        return self.observation, self.reward

    def get_observation(self):
        if self.flag:
            self.observation = np.array([float(self.taxi_loc)/16, float(self.up_loc)/16])
        else:
            self.observation = np.array([float(self.taxi_loc)/16, float(self.down_loc)/16])
