import random
from collections import deque
import numpy as np
import tensorflow as tf


class Agent:
    def __init__(
            self,
            n_features=2,
            n_actions=16,
            learning_rate=0.01,
            batch_size=32,
            replace_target_iter=300,
            reward_decay=0.9,
            e_greedy=0.9,
            e_greedy_increment=None,
            memory_size=40
    ):
        # 输入参数
        self.n_features = n_features
        self.n_actions = n_actions
        # 神经网络和DQN参数
        self.lr = learning_rate
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        # 学习步数
        self.learn_step_counter = 0
        # 初始化memory pool [s, a, r, s_]
        self.memory_counter = 0
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        # evaluate_net和target_net
        self.evaluate_net = self.build_net()
        self.target_net = self.build_net()
        self.connect = np.loadtxt('./connect.csv', delimiter=',', dtype=int)

    def build_net(self):
        model = tf.keras.models.Sequential([
                  tf.keras.layers.Flatten(input_shape=(self.n_features,)),
                  tf.keras.layers.Dense(256, activation='relu'),
                  # tf.keras.layers.Dense(256, activation='relu'),
                  # tf.keras.layers.Dense(256, activation='relu'),
                  tf.keras.layers.Dense(128, activation='relu'),
                  tf.keras.layers.Dropout(0.2),
                  tf.keras.layers.Dense(self.n_actions, activation='linear')
                ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def save_model(self, episode):
        self.target_net.save(str(episode)+'.h5')
        print("Target_net model saved to h5 file.")

    def store_transition(self, s, a, r, s_):
        self.memory.append((s, a, r, s_))
        self.memory_counter += 1

    def choose_action(self, observation):
        if np.random.uniform() < self.epsilon:
            observation_reshaped = tf.reshape(observation, [-1, 2])
            actions_value = self.evaluate_net.predict(observation_reshaped)
            action = np.argmax(actions_value[0])
        else:
            action = np.random.randint(0, self.n_actions)  # 完全随机选择
        return int(action)

    def learn(self):
        # memory pool 未存满则先返回
        if self.memory_counter < self.memory_size:
            return

        # 检查是否需要更新 target_net 参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.set_weights(self.evaluate_net.get_weights())
            print('\ntarget_params_replaced\n')

        # 从 memory pool 中采样
        batch_memory = random.sample(self.memory, self.batch_size)
        batch_s = []
        batch_s_ = []
        for replay in batch_memory:
            batch_s.append(replay[0])
            batch_s_.append(replay[3])
        q_eval = self.evaluate_net.predict(batch_s)
        q_next = self.target_net.predict(batch_s_)
        # print("before, q_eval=", q_eval)
        # print("before, q_next=", q_next)
        # 使用公式更新Q值
        for i, replay in enumerate(batch_memory):
            _, a, reward, _ = replay
            q_eval[i][a] = (1 - self.lr) * q_eval[i][a] + self.lr * (reward + self.gamma * np.amax(q_next[i]))

        # print("after, q_eval_=", q_eval)
        # 训练 evaluate_net
        batch_s_shaped = tf.reshape(batch_s, [-1, 2])
        self.evaluate_net.fit(batch_s_shaped, q_eval)
        # print(batch_s_shaped)
        # print("\n")
        # print(q_eval)
        # print("\n")

        # 增加 epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
