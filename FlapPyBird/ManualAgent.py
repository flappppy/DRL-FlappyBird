from Agent import Agent
import pygame
from pygame.locals import *


class ManualAgent(Agent):
    def __init__(self):
        self.count=0
        return

    def getNextAction(self, state):
        self.count+=1
        eventList= pygame.event.get()
        res=[]
        for event in eventList:
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                res.append(-1)
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                res.append(1)
        return res


