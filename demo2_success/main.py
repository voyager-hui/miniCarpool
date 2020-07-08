from RL_brain import DeepQNetwork
from env import Env


if __name__ == "__main__":
    ENV = Env()
    RL = DeepQNetwork(4, 2,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )

    step = 1
    for episode in range(10000):
        ENV.reset()  # 重置环境
        print("episode:", episode)
        while True:
            observation = ENV.refresh_env()
            action1 = RL.choose_action(observation[0, :])
            action2 = RL.choose_action(observation[1, :])
            observation_, reward = ENV.taxi_refresh([action1, action2])
            RL.store_transition(observation[0, :], action1, reward[0], observation_[0, :])
            RL.store_transition(observation[1, :], action2, reward[1], observation_[1, :])
            print("taxi1", ENV.taxi1_loc, "taxi2", ENV.taxi2_loc)
            if step % 20 == 0:
                RL.learn()
            if ENV.done:
                break
            step += 1
        # if episode >= 1000 and episode % 1000 == 0:
        #     RL.save_model(episode)  # 保存神经网络参数
    RL.plot_cost()