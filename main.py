from agent import Agent
from env import Env


if __name__ == "__main__":
    ENV = Env()
    RL = Agent()

    step = 1
    for episode in range(10000):
        ENV.reset()  # 重置环境
        print("episode:", episode)
        while True:
            observation = ENV.refresh_env()  # 获取环境信息
            action = RL.choose_action(observation)  # 神经网络根据信息做决策
            observation_, reward = ENV.taxi_refresh(action)  # 执行决策并获取环境反馈
            if reward >= 0:
                RL.store_transition(observation, action, reward, observation_)  # 存储有效经历
            print(ENV.taxi_loc)
            if step % 20 == 0:
                RL.learn()  # 通过经历训练神经网络
            if ENV.done:
                break  # 退出本episode
            step += 1
        if episode >= 1000 and episode % 1000 == 0:
            RL.save_model(episode)  # 保存神经网络参数
