import random


class Action():
    def __init__(self, typed=0):
        self.oper = ["NoOperation", "Migrate"]
        self.dst = []
        if isinstance(typed, int):
            assert (typed < len(self.oper)), "Code operation not valid"
            self.type = typed
        elif isinstance(typed, str):
            try:
                self.type = self.oper.index(typed)
            except ValueError:
                raise Exception("Not valid operation")

    def __repr__(self):
        return self.oper[self.type]

    def __str__(self):
        return self.oper[self.type]

    def __eq__(self, a2):
        return self.type == a2.type

    def __len__(self):
        return len(self.oper)

    def sample(self):
        return Action(random.randint(0, len(self.oper) - 1))
