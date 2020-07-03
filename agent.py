import random
from collections import deque
import numpy as np
import tensorflow as tf
from my_network import DQNNetwork


class Agent:
    def __init__(
            self,
            n_features=2,
            n_actions=16,
            IMG_HEIGHT=300,
            IMG_WIDTH=300,
            learning_rate=0.01,
            batch_size=32,
            replace_target_iter=300,
            reward_decay=0.9,
            e_greedy=0.9,
            e_greedy_increment=0.01,
            memory_size=40
    ):
        # 输入参数
        self.n_features = n_features
        self.n_actions = n_actions
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        # 神经网络和DQN参数
        self.lr = learning_rate
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.0 if e_greedy_increment is not None else self.epsilon_max
        # 学习步数
        self.learn_step_counter = 0
        # 初始化memory pool [s, a, r, s_]
        self.memory_counter = 0
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        # evaluate_net和target_net
        self.evaluate_net = DQNNetwork(n_features=self.n_features,
                                       n_actions=self.n_actions)
        self.target_net = DQNNetwork(n_features=self.n_features,
                                     n_actions=self.n_actions)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-6)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.connect = np.loadtxt('./connect.csv', delimiter=',', dtype=int)

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

        # 从 memory pool 中采样并解析S,A,R,S_
        batch_memory = random.sample(self.memory, self.batch_size)
        batch_s = []
        batch_action = []
        batch_reward = []
        batch_s_ = []
        for replay in batch_memory:
            batch_s.append(replay[0])
            batch_action.append(replay[1])
            batch_reward.append(replay[2])
            batch_s_.append(replay[3])
        batch_s = tf.reshape(batch_s, [-1, 2])
        batch_action = tf.reshape(batch_action, [-1, ])
        batch_reward = tf.reshape(batch_reward, [-1, ])
        batch_s_ = tf.reshape(batch_s_, [-1, 2])
        # 使用公式更新Q值
        with tf.GradientTape() as tape:
            q_online = self.evaluate_net.predict(batch_s_)
            action_q_online = tf.math.argmax(q_online, axis=1)
            q_target = self.target_net.predict(batch_s_)
            ddqn_q = tf.reduce_sum(q_target * tf.one_hot(action_q_online, self.n_actions, 1.0, 0.0), axis=1)
            expected_q = batch_reward + 0.99 * ddqn_q
            main_q = tf.reduce_sum(self.evaluate_net.predict(batch_s) * tf.one_hot(batch_action, self.n_actions, 1.0, 0.0), axis=1)
            loss = self.loss(tf.stop_gradient(expected_q), main_q)
            tape.watch(self.evaluate_net.trainable_variables)
            print("---------------------------------------------------")
            print("main_q", main_q)
            print("expected_q", expected_q)
            print("loss", loss)
        # 训练 evaluate_net
        gradients = tape.gradient(loss, self.evaluate_net.trainable_variables)
        print("gradients", gradients)
        self.optimizer.apply_gradients(zip(gradients, self.evaluate_net.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(expected_q, main_q)

        # 增加 epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
