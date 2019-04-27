import json

class Agent:
    def __init__(self):
        self.dumpNumber = 30
        self.gameCount = 0
        self.reward = [1, -1000]
        self.discount = 1.0
        self.learningRate = 0.5
        self.load_qvalues()
        self.lState = "420_240_0"
        self.lAction = 0
        self.actions = []

    def load_qvalues(self):
        self.qvalues = {}
        f = open("qvalues.json", "r")
        self.qvalues = json.load(f)
        f.close()

    def dump_qvalues(self, exit = False):
        if self.gameCount % self.dumpNumber == 0 or exit is True:
            f = open("qvalues.json", "w")
            json.dump(self.qvalues, f)
            f.close()

    def act(self, cs):
        if self.qvalues[cs][0] >= self.qvalues[cs][1]:
            self.lAction = 0
            return 0
        else:
            self.lAction = 1
            return 1

    def update(self, dump_qvalues = True):
        history = list(reversed(self.actions))

        flag = True if int(history[0][2].split("_")[1]) > 120 else False

        t = 1
        for i in history:
            state = i[0]
            act = i[1]
            estimate = i[2]

            if t == 1 or t == 2:
                reward = self.reward[1]
            elif flag and act:
                reward = self.reward[1]
                flag = False
            else:
                reward = self.reward[0]

            self.qvalues[state][act] = (1-self.learningRate) * (self.qvalues[state][act]) + \
                                       self.learningRate * ( reward + self.discount * max(self.qvalues[estimate]) )
            t += 1

        self.gameCount += 1
        if dump_qvalues:
            self.dump_qvalues()
        self.actions = []

    def stateToGrid(self,x, y, x1, y1, x2, y2, v):
        gridX = (x1 if(x < 140) else x2)
        gridY = (y1 if (y < 180) else y2)

        girdState = str(int(gridX)) + "_" + str(int(gridY)) + "_" + str(v)

        return girdState
