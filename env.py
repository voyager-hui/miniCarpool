class Env:
    def __init__(self, run_time=300):
        self.done = False  # 结束标志（current_time>=run_time）
        self.run_time = run_time  # 模拟运行总时间（单位分钟）
        self.current_time = -1  # 当前时刻（单位分钟）
        self.observation = []  # 当前环境的
        self.taxi_loc = 0  # taxi的位置坐标
        self.myreward = 0

    def reset(self, run_time=300):
        self.done = False  # 结束标志（current_time>=run_time）
        self.run_time = run_time  # 模拟运行总时间（单位分钟）
        self.current_time = -1  # 当前时刻（单位分钟）
        self.observation = []  # 当前环境的状态
        self.taxi_loc = 0  # taxi的位置坐标
        self.myreward = 0

    def refresh_env(self):
        # 时间流逝一分钟
        self.current_time += 1
        if self.current_time >= self.run_time:
            self.done = True
        self.get_observation()
        return self.observation

    # 执行决策
    def taxi_refresh(self, action):
        self.myreward = 0
        # 更新出租车位置
        if action == 0:
            if self.taxi_loc != 0:
                self.taxi_loc -= 1
            else:
                self.myreward = -1
        else:
            self.taxi_loc += 1
        # 是否到达目标点
        if self.taxi_loc == 6:
            self.done = True
            self.myreward = 1
        self.get_observation()
        return self.observation, self.myreward

    # 生成 observation
    def get_observation(self):
        self.observation = [self.taxi_loc]
