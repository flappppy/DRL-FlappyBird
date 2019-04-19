import gym
from keras import models
from keras import layers
from keras.optimizers import Adam
from collections import deque
import random
import numpy as np


#############################
# if you want to use GPU to boost, use these code.

# import tensorflow as tf
# import keras
# config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 1} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)

#############################


class DeepTrain:
    def __init__(self):
        self.stateShape=4

        self.actionSpaceNum=2

        self.gamma = 0.99

        self.epsilon = 0.2
        self.epsilon_decay = 0.001

        self.epsilon_min = 0.0001

        self.learingRate = 0.0001

        self.replayBuffer = deque(maxlen=60000)
        self.trainNetwork = self.createNetwork()

        self.episodeNum = 400

        self.iterationNum = 201  # max is 200

        self.numPickFromBuffer = 32*2

        self.targetNetwork = self.createNetwork()

        self.targetNetwork.set_weights(self.trainNetwork.get_weights())


    def createNetwork(self):
        model = models.Sequential()

        model.add(layers.Dense(32,activation='relu', input_shape=(self.stateShape,)))
        model.add(layers.Dense(64,activation='relu'))
        model.add(layers.Dense(self.actionSpaceNum, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learingRate))
        return model

    def getBestAction(self, state):

        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.rand(1) < self.epsilon:
            value = np.random.randint(0, 100)
            if value>80:
                action=1
            else:
                action=0
        else:
            pre=self.trainNetwork.predict(state)
            action = np.argmax(pre[0])

        return action

    def trainFromBuffer_Boost(self):
        if len(self.replayBuffer) < self.numPickFromBuffer:
            return
        samples = random.sample(self.replayBuffer, self.numPickFromBuffer)
        npsamples = np.array(samples)
        states_temp, actions_temp, rewards_temp, newstates_temp, dones_temp = np.hsplit(npsamples, 5)
        states = np.concatenate((np.squeeze(states_temp[:])), axis=0)
        rewards = rewards_temp.reshape(self.numPickFromBuffer, ).astype(float)
        targets = self.trainNetwork.predict(states)
        newstates = np.concatenate(np.concatenate(newstates_temp))
        dones = np.concatenate(dones_temp).astype(bool)
        notdones = ~dones
        notdones = notdones.astype(float)
        dones = dones.astype(float)
        Q_futures = self.targetNetwork.predict(newstates).max(axis=1)
        targets[(np.arange(self.numPickFromBuffer),
                 actions_temp.reshape(self.numPickFromBuffer, ).astype(int))] = rewards * dones + (
                    rewards + Q_futures * self.gamma) * notdones
        self.trainNetwork.fit(states, targets, epochs=1, verbose=0)

    def trainFromBuffer(self):
        if len(self.replayBuffer) < self.numPickFromBuffer:
            return

        samples = random.sample(self.replayBuffer, self.numPickFromBuffer)

        states = []
        newStates = []
        for sample in samples:
            state, action, reward, new_state, done = sample
            states.append(state)
            newStates.append(new_state)

        newArray = np.array(states)
        states = newArray.reshape(self.numPickFromBuffer, self.stateShape)

        newArray2 = np.array(newStates)
        newStates = newArray2.reshape(self.numPickFromBuffer, self.stateShape)

        targets = self.trainNetwork.predict(states)
        new_state_targets = self.targetNetwork.predict(newStates)

        i = 0
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = targets[i]
            if done:
                target[action] = reward
            else:
                Q_future = max(new_state_targets[i])
                target[action] = reward + Q_future * self.gamma
            i += 1



        self.trainNetwork.fit(states, targets, epochs=1, verbose=0)

    def orginalTry(self, currentState, eps):
        rewardSum = 0
        max_position = -99

        for i in range(self.iterationNum):
            bestAction = self.getBestAction(currentState)

            # show the animation every 50 eps
            if eps % 50 == 0:
                env.render()

            new_state, reward, done, _ = env.step(bestAction)

            new_state = new_state.reshape(1, 2)

            # # Keep track of max position
            if new_state[0][0] > max_position:
                max_position = new_state[0][0]

            # # Adjust reward for task completion
            if new_state[0][0] >= 0.5:
                reward += 10

            self.replayBuffer.append([currentState, bestAction, reward, new_state, done])

            # Or you can use self.trainFromBuffer_Boost(), it is a matrix wise version for boosting
            self.trainFromBuffer()

            rewardSum += reward

            currentState = new_state

            if done:
                break

        if i >= 199:
            print("Failed to finish task in epsoide {}".format(eps))
        else:
            print("Success in epsoide {}, used {} iterations!".format(eps, i))
            self.trainNetwork.save('./trainNetworkInEPS{}.h5'.format(eps))

        # Sync
        self.targetNetwork.set_weights(self.trainNetwork.get_weights())

        print("now epsilon is {}, the reward is {} maxPosition is {}".format(max(self.epsilon_min, self.epsilon),
                                                                             rewardSum, max_position))
        self.epsilon -= self.epsilon_decay


    def trainOnce(self,currentState):
        return

    def start(self):
        for eps in range(self.episodeNum):
            currentState = env.reset().reshape(1, 2)
            self.orginalTry(currentState, eps)



