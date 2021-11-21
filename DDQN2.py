# Sevgilime Notlar:
# rewardü kademeli olarak arttırmak.
# Sunuma 47. satırı koy statelerdeki pixel değişimleri yılana göre.
import tensorflow as tf
import os
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, \
    ZeroPadding2D, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import plot_model
import numpy as np

tf.keras.backend.clear_session()

tf.compat.v1.disable_eager_execution()

hello = tf.constant('Hello, TensorFlow!')

sess = tf.compat.v1.Session()

print(sess.run(hello))


class ReplayBuffer:

    def __init__(self, memory_size, n_action, input_dims):
        self.memory_size = memory_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, input_dims[0], input_dims[1], input_dims[2]))
        self.new_state_memory = np.zeros((self.memory_size, input_dims[0], input_dims[1], input_dims[2]))
        self.action_memory = np.zeros((self.memory_size, n_action), dtype=np.int8)
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size)

    def store_transition(self, state_flash, action, reward, state_, done):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state_flash
        self.new_state_memory[index] = state_
        actions = np.zeros(self.action_memory.shape[1])
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)

        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, new_states, actions, rewards, terminal


def build_ddqn(lr, n_actions, conv1_dims, conv2_dims, input_dims):
    model = Sequential([
        ZeroPadding2D(padding=(4, 4), input_shape=input_dims),
        Conv2D(conv1_dims, kernel_size=(8, 8), strides=(4, 4), padding="same"),
        # BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(4, 4), strides=(2, 2), padding='same'),
        ZeroPadding2D(padding=(2, 2)),
        Conv2D(conv2_dims, kernel_size=(4, 4), strides=(2, 2), padding="same"),
        # BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        Flatten(),
        Dense(n_actions, activation='relu')
    ])

    model.compile(optimizer=Adam(lr=lr), loss='mse')
    model.summary()
    return model

# def build_ddqn(lr, n_actions, conv1_dims, conv2_dims, input_dims):
#     model = Sequential([
#         ZeroPadding2D(padding=(2, 2)),
#         Conv2D(conv1_dims, kernel_size=(4, 4), strides=(2, 2), padding="valid", input_shape=input_dims),
#         BatchNormalization(),
#         Activation('relu'),
#         MaxPooling2D(pool_size=(4, 4), strides=(2, 2), padding='same'),
#         ZeroPadding2D(padding=(1, 1)),
#         Conv2D(conv2_dims, kernel_size=(2, 2), strides=(1, 1), padding="valid"),
#         BatchNormalization(),
#         Activation('relu'),
#         MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
#         Flatten(),
#         Dense(256, activation='relu'),
#         Dense(n_actions, activation='relu')
#     ])
#
#     model.compile(optimizer=Adam(lr=lr), loss='mse')
#     return model


class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, batch_size, input_dims,
                 epsilon=1.0, epsilon_dec=(1 * 10 ** (-5)), epsilon_end=0.0001,
                 mem_size=50000, fname='ddqn_model.h5', calistir=1,
                 observe=2000, replace_target=1000):
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.gamma = gamma
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.alpha = alpha
        self.model = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, n_actions, input_dims)
        self.q_eval = build_ddqn(alpha, n_actions, 32, 64, input_dims)
        self.q_target = build_ddqn(alpha, n_actions, 32, 64, input_dims)
        self.t = 0
        self.loss = np.array([])
        self.i = 0
        self.loss_counter = 0
        self.epsilon = epsilon
        self.observe = observe
        self.calistir = calistir
        self.is_train_and_weight()
        self.run_target = False
        self.epsilon_counter = 0
        self.episode_counter = 1

    def is_train_and_weight(self):
        if self.calistir == 1:
            pass
        elif self.calistir == 2:
            self.epsilon = self.epsilon_min
            self.observe = 0
        elif self.calistir == 3:
            self.epsilon = self.epsilon_min
            self.observe = 0
        else:
            raise Exception('"calistir" parameter is wrong')

    def remember(self, state_flash, action, reward, new_state, done):
        self.memory.store_transition(state_flash, action, reward, new_state, done)

    def choose_action(self, state_flash):
        if self.t < self.observe:
            action = np.random.choice(self.action_space)
        else:
            if np.random.random() < self.epsilon:
                action = np.random.choice(self.action_space)
            else:
                if self.calistir == 2 or self.calistir == 3:
                    self.load_model()

                actions = self.q_eval.predict(state_flash)
                action = np.argmax(actions)

        self.t += 1
        return action

    def learn(self):
        if self.memory.memory_counter > self.batch_size:
            state, new_state, action, reward, done = self.memory.sample_buffer(self.batch_size)
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)
            action_indices = action_indices.astype(int)

            if self.run_target or (self.epsilon == self.epsilon_min):
                q_next = self.q_target.predict(new_state)
                # print("epsilonmine ulaştı")
            # q_next = np.interp(q_next, (q_next.min(), q_next.max()), (-1, +1))

            q_eval = self.q_eval.predict(new_state)
            # q_eval = np.interp(q_eval, (q_eval.min(), q_eval.max()), (-1, +1))
            # q_next = q_eval
            if self.run_target == False:
                if self.calistir == 1:
                    q_next = q_eval
                    self.epsilon_counter = 1 if self.epsilon == self.epsilon_min and self.epsilon_counter == 0 else 0
                    # print("başlangıç")
                else:
                    self.run_target = True
                    self.epsilon_counter = 1 if self.epsilon == self.epsilon_min and self.epsilon_counter == 0 else 0
                self.run_target = True if self.epsilon == self.epsilon_min else False

            q_target = self.q_eval.predict(state)
            # q_pred = np.interp(q_pred, (q_pred.min(), q_pred.max()), (-1, +1))

            max_actions = np.argmax(q_eval, axis=1)

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + self.gamma * q_next[
                batch_index, max_actions.astype(int)] * done

            # # print(q_target)
            # # print("+++")
            # # self.scaler.fit(q_target)
            # # q_target = self.scaler.transform(q_target)
            # # print(q_target)
            # # print("---")
            # # print(q_target/np.max(abs(q_target)).astype(np.float))
            # # print("***")
            # q_target = q_target/np.max(abs(q_target)).astype(np.float)

            # self.loss = np.append(self.loss,self.q_eval.train_on_batch(state, q_target))

            _ = self.q_eval.train_on_batch(state, q_target)
            # self.q_eval.summary()
            # plot_model(q_eval, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
            self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

            if self.epsilon_counter == 1:
                self.q_target.predict(new_state)
                self.uptade_network_parameters()
                self.epsilon_counter += 1
            if (self.episode_counter % self.replace_target == 0) and self.run_target:
                # print("10 episode tamamlandı")
                # self.uptade_network_parameters()
                # self.uptade_network_parameters()
                # # self.q_target.save_weights('ddqn_model.hdf5', overwrite=True)
                # # model_json = self.q_target.to_json()
                # # with open("model.json", "w") as json_file:
                # #     json_file.write(model_json)
                # # serialize weights to HDF5
                # # f_name = 'ddqn_model.h5'
                #
                # self.i += 1
                # self.save_network_parameters()
                self.uptade_network_parameters()
                self.save_model()
                self.episode_counter += 1

    def save_network_parameters(self):
        self.q_target.save_weights(self.model, overwrite=True)

    def uptade_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    # Weights dosyası yükleme
    def load_weight(self):
        if self.t == 33:
            if self.calistir == 2:
                self.q_eval.load_weights(self.model)
                self.q_target.load_weights(self.model)
                print(str(self.model) + " dosyası yüklendi. Train oluyor.")
                os.rename(self.model, self.model + '.backup')
                if self.epsilon <= self.epsilon_min:
                    self.uptade_network_parameters()
            elif self.calistir == 3:
                self.q_eval.load_weights(self.model)
                print(str(self.model) + " dosyası yüklendi. Train olmayacak")

    def save_model(self):
        self.q_eval.save(self.model)

    def load_model(self):
        if self.t == 1:
            if self.calistir == 2:
                self.q_eval = load_model(self.model)
                self.q_target = load_model(self.model)
                print(str(self.model) + " dosyası yüklendi. Train oluyor.")
                os.rename(self.model, self.model + '.backup')
                if self.epsilon <= self.epsilon_min:
                    self.uptade_network_parameters()
            elif self.calistir == 3:
                self.q_eval = load_model(self.model)
                print(str(self.model) + " dosyası yüklendi. Train olmayacak")
