from Agent import Agent
import pygame
from pygame.locals import *

import time


class SimplePolicyAgent(Agent):
    def __init__(self):
        self.count=0
        return


    # touch the screen every 15 function call
    def getNextAction(self, state):
        self.count+=1
        if self.count%15==1:
            self.generateEvent(1)
        eventList = pygame.event.get()

        res = []
        for event in eventList:
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                res.append(-1)
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                res.append(1)
        return res
