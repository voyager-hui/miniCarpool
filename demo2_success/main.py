from RL_brain import DeepQNetwork
from env import Env
from visual import read_log
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    if os.path.exists("log.txt"):
        os.remove("log.txt")
    ENV = Env()
    RL = DeepQNetwork(4, 2,
                      learning_rate=0.01,
                      reward_decay=0.75,
                      e_greedy=0.8,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

    step = 1
    reward_his = []
    for episode in range(200):
        ENV.reset()  # 重置环境
        re_hi = 0.0
        # print("episode:", episode)
        with open('log.txt', 'a+') as f:
            f.write("episode:"+str(episode)+"\n")
        while True:
            observation = ENV.refresh_env()  # 获取环境信息
            action = RL.choose_action(observation)  # 神经网络根据信息做决策
            observation_, reward = ENV.taxi_refresh(action)  # 执行决策并获取环境反馈
            RL.store_transition(observation, action, reward, observation_)  # 存储经历
            re_hi += reward
            # print(ENV.up_loc, " ", ENV.down_loc, " ", ENV.taxi_loc)
            with open('log.txt', 'a+') as f:
                f.write(str(ENV.up_loc)+" "+str(ENV.down_loc)+" "+str(ENV.taxi_loc)+"\n")
            if step % 20 == 0:
                RL.learn()  # 通过经历训练神经网络
            if ENV.done:
                break  # 退出本episode
            step += 1
        reward_his.append(re_hi)
    with open('log.txt', 'a+') as f:
        f.write("end")
    # 绘制loss
    RL.plot_cost()
    # 绘制reward
    plt.plot(np.arange(len(reward_his)), reward_his)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.show()
    # 可视化动画
    read_log('log.txt')
