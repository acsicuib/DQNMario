import random


class Action():
    def __init__(self, typed=0):
        self.oper = ["NoOperation", "Migrate"]
        self.dst = [] # DST operation using ID - node
        self.relative_dst = [] # DST operation relative POS in edge-nodes (neighbour nodes)

        if isinstance(typed, int):
            assert (typed < len(self.oper)), "Code operation not valid"
            self.code = typed
        elif isinstance(typed, str):
            try:
                self.code = self.oper.index(typed)
            except ValueError:
                raise Exception("Not valid operation")

    def __repr__(self):
        return self.oper[self.code]

    def __str__(self):
        return "%s: %s %s" % (self.oper[self.code], self.dst, self.relative_dst)

    def __eq__(self, a2):
        return self.code == a2.code

    def __len__(self):
        return len(self.oper)

    def sample(self):
        return Action(random.randint(0, len(self.oper) - 1))

