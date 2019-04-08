import flappy
import pygame
from pygame.locals import *

class Agent:

    def play(self):
        flappy.startWithAgent(self)


    # action leads to the new state
    #oldState -> action -> newState
    def onStateChange(self,oldState, action, newState):
        print(str(newState)+" -->"+str(action)+" -->"+str(newState))

    def getNextAction(self,state):
        return pygame.event.get()

    def generateEvent(self, signal):
        if signal==1:
            e=pygame.event.Event(KEYDOWN, {"key": K_SPACE, "mod": 0, "unicode": u' '})
            pygame.event.post(e)
        if signal==-1:
            e = pygame.event.Event(KEYDOWN, {"key": K_ESCAPE, "mod": 0, "unicode": u' '})
            pygame.event.post(e)




