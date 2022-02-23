import random
import numpy as np

ACTION_NAMES = ["NoOperation", "Migrate"]

class Action():

    def __init__(self, size, typed=0):
        self.action = np.zeros(size)
        self.size = size
        self.dst = [] # DST operation using ID - node
        self.relative_dst = [] # DST operation relative POS in edge-nodes (neighbour nodes)

        if isinstance(typed, int):
            assert (typed < len(ACTION_NAMES)), "Code operation not valid"
            self.code = typed
        elif isinstance(typed, str):
            try:
                self.code = ACTION_NAMES.index(typed)
            except ValueError:
                raise Exception("Not valid operation")

    def __repr__(self):
        return self.action

    def __str__(self):
        return "A[%s %s: %s %s]" % (self.action, ACTION_NAMES[self.code], self.dst, self.relative_dst)

    def __eq__(self, a2):
        if isinstance(a2, str):
            if a2 not in ACTION_NAMES:
                raise Exception("Code string not valid")
            code = ACTION_NAMES.index(a2)
            return self.code == code
        else:
            return self.code == a2.code

    def __len__(self):
        return len(ACTION_NAMES)

def action_sample(size):
    type = random.randint(0, len(ACTION_NAMES) - 1)
    a = Action(size,type)
    if type == 0:
        return a
    else:
        pos = np.random.choice(size,1)[0]
        a.action[pos]=type
        return a

