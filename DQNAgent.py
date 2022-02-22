import random
from collections import deque

import numpy as np
import torch
from numpy.random import RandomState
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from action import Action
from models import MLP


class Agent():

    def __init__(self, num_node_features, num_classes):
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.rnd = RandomState(0)  # seed
        self.batch_size = 3
        self.memory = deque(maxlen=50)
        self.model = MLP(num_node_features=num_node_features, hidden_channels=8, num_classes=num_classes)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_op = torch.nn.CrossEntropyLoss()

    def act(self, state):
        v = self.rnd.rand()
        if v <= self.epsilon:
            return self.pickOne(state)
        else:
            # TODO
            # generate data.x

            # _, pred = self.model(data.x, data.edge_index).max(dim=1)

            # print(pred)
            return self.pickOne(state) #TEST
            # TODO from pred get action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def __get_y_vector(self,shape,action):
        y = np.zeros(shape[0])
        if action == Action("Migrate"):
            y[action.relative_dst[0]] = 1
        return y

    def __get_loader_states(self, samples):
        data_list = []
        for state, action, *rest in samples:
            y = self.__get_y_vector(state["feat"].shape,action)
            data = Data(x=state["feat"], edge_index=state["edges"], y=y)
            data_list.append(data)
        loader = DataLoader(data_list, batch_size=self.batch_size)
        return loader

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        non_final_next_states = [(state, action, reward, state_next, done)
                                 for state, action, reward, state_next, done in samples
                                 if done is False]

        loader_states = self.__get_loader_states(samples)
        # actions = self.__get_actions(samples)

        # TODO
        #         # Organize the data?
        #         #for state, action, reward, next_state, done in minibatch:

        #         #for data in train_loader:
        #             data = data.to(device)
        #             self.optimizer.zero_grad()
        #             output = self.model(data.x, data.edge_index,data.batch) #.max(dim=1)
        #             loss = self.loss_op(output, data.y)
        #             loss.backward()
        #             total_loss += data.num_graphs * loss.item()
        #             self.optimizer.step()
        #             epoch_loss = total_loss / len(train_loader.dataset)
        #             #writer.add_scalar('Loss/train', epoch_loss, global_step = epoch)

        #         # Keeping track of loss
        #         loss = history.history['loss'][0]
        #         if self.epsilon > self.epsilon_min:
        #             self.epsilon *= self.epsilon_decay
        #         return loss
        return None

    def pickOne(self, state):  # Do a random action
        a = Action().sample()
        if a == Action("Migrate"):
            neighs = state["feat"][:, 0][state["feat"][:, 0] >= 0]
            assert len(neighs) > 0
            a.dst = np.random.choice(neighs[1:], 1)
            a.relative_dst = np.where(neighs==a.dst[0])[0]
            assert len(a.dst)==len(a.relative_dst)
        return a
