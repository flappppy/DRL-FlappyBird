from Agent import Agent
import pygame
from pygame.locals import *
from DeepTrain import DeepTrain
import numpy as np
from keras import models

class PredictAgent(Agent):
    def __init__(self):
        self.model=DeepTrain()

        self.syncCount=0

        self.tryTime=0

        self.model=self.loadModel()

        return

    def loadModel(self):
        # return models.load_model('../bestModel/trainNetworkInEPS682.h5')
        return models.load_model('../bestModel/trainNetworkInEPS788.h5')


    def onStateChange(self, oldState, action, newState):
        self.syncCount+=1
        # print(str(oldState) + " -->" + str(action) + " -->" + str(newState))
        return

    def getActionFromModel(self,state):
        pre=self.model.predict(state)
        action=np.argmax(pre[0])
        return action

    def getNextAction(self, state):
        new_state, score, done=self.processState(state)

        action=self.getActionFromModel(np.reshape(new_state, (4, -1)).T)
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