from Agent import Agent
import pygame
from pygame.locals import *
from DeepTrain import DeepTrain
import numpy as np


class DeepAgent(Agent):
    def __init__(self):
        self.model=DeepTrain()

        self.syncCount=0

        self.tryTime=0

        self.lifeTime=0

        self.postiveList=[]

        self.exploreTime=0

        return

    def onStateChange(self, oldState, action, newState):
        self.syncCount+=1
       # print(str(oldState) + " -->" + str(action) + " -->" + str(newState))

        #process action
        if len(action)==0:
            action=0
        else:
            action=action[0]

        currentState, currentScore, currentDone=self.processState(oldState)

        newState, newScore, newDone = self.processState(newState)

        # reward=self.lifeTime
        reward= newScore


        if newState[0][1]>250:
            reward-=200


        #extra bouns
        # if newScore>0:
        #     reward+=10000
        #     reward+=newScore**2

        if newDone:
            self.onDone(reward)
            self.model.replayBuffer.append([currentState, action, -1000+newScore, newState, newDone])
            print("Try "+str(self.tryTime)+" finished, the score is:"+str(newScore)+",epslion="+str(max(self.model.epsilon_min, self.model.epsilon)))
            # print("reward: -1000")
            return
            # print("reward: "+str(reward))


        self.model.replayBuffer.append([currentState, action, reward, newState, newDone])
        if self.exploreTime >= 10000:
            self.model.trainFromBuffer()

        #good model
        if newScore>=10 and newScore%10==0:
            self.model.trainNetwork.save('./trainNetworkInEPS{}.h5'.format(self.tryTime))

        return



    def getNextAction(self, state):
        new_state, score, done=self.processState(state)
        if self.exploreTime<10000:
            self.exploreTime+=1
            #pure explore
            value = np.random.randint(0, 100)
            if value > 90:
                action = 1
            else:
                action = 0
        else:
            action = self.model.getBestAction(np.reshape(new_state, (4, -1)).T)
        # print("action"+str(action))
        self.generateEvent(action)

        eventList = pygame.event.get()
        res = []
        for event in eventList:
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                res.append(-1)
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                res.append(1)
        return res



#[xdiff,ydiff,'playerVelY','playerRot']
    def processState(self,rawState):
        new_state=[]

        # TODO  pick states
        lowerPipes=rawState["lowerPipes"]
        playerx=rawState["playerx"]
        playery=rawState["playery"]

        if lowerPipes[0]['x']-playerx >-30:
            new_state.append(lowerPipes[0]["x"]-playerx)
            new_state.append(lowerPipes[0]["y"]-playery)
        else:
            new_state.append(lowerPipes[1]["x"]-playerx)
            new_state.append(lowerPipes[1]["y"]-playery)

        new_state.append(rawState["playerVelY"])
        new_state.append(rawState["playerRot"])

        score=rawState["score"]
        done=rawState["isDead"]

        new_state=np.asarray(new_state)

        new_state = new_state.reshape(1, 4)

        return  new_state, score, done

    def onDone(self,reward):
        self.tryTime += 1
        self.lifeTime = 0
        self.model.targetNetwork.set_weights(self.model.trainNetwork.get_weights())
        if reward>0:
            self.model.epsilon -= self.model.epsilon_decay



