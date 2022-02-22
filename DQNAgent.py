import random
from collections import deque

import numpy as np
import torch
from numpy.random import RandomState
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

from action import Action
from models import MLP

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor



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
            node_feats = torch.tensor(state["feat"], dtype=torch.float)
            edge_index = torch.tensor(state["edges"], dtype=torch.long)
            y = torch.tensor(y, dtype=torch.int64)
            data = Data(x=node_feats, edge_index=edge_index, y=y)
            data_list.append(data)
        loader = DataLoader(data_list, batch_size=self.batch_size)
        return loader

    # def __get_loader_states(self, samples):
    #     data_list = []
    #     for _, _, _, next_state,_ in samples:
    #         y = self.__get_y_vector(next_state["feat"].shape,action)
    #         node_feats = torch.tensor(next_state["feat"], dtype=torch.float)
    #         edge_index = torch.tensor(next_state["edges"], dtype=torch.long)
    #         y = torch.tensor(y, dtype=torch.int64)
    #         data = Data(x=node_feats, edge_index=edge_index, y=y)
    #         data_list.append(data)
    #     loader = DataLoader(data_list, batch_size=self.batch_size)
    #     return loader

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)

        non_final_next_states = [(s,a,r,ns,done) for (s,a,r,ns,done) in samples if done is False]
        print(len(samples))
        print(len(non_final_next_states))
        non_final_next_states = list(self.__get_loader_states(non_final_next_states))[0]
        print(non_final_next_states)
        #El problema es que el non_final_maks debería de tener 4 (salidas) por cada done
        non_final_mask = [int(done) for *rest, done in samples ] #could be improved... * *features
        non_final_mask = ByteTensor(non_final_mask)

        loader_states = self.__get_loader_states(samples)
        data =  list(loader_states)[0] #loader has len batch_size
        output, state_action_values  = self.model(data.x, data.edge_index,data.batch).max(dim=1)

        next_state_values = Variable(torch.zeros(self.batch_size).type(Tensor))

        print(next_state_values)
        print(non_final_mask)
        print(self.model(non_final_next_states.x,non_final_next_states.edge_index).max(dim=1)[0])
        next_state_values[non_final_mask] =  self.model(non_final_next_states.x,non_final_next_states.edge_index).max(dim=1)


        # print(data)
        # print("P:",pred)
        # print("Y:",data.y)
        # print("-" * 10)

        # # Compute the expected Q values
        # expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        #
        # # Compute Huber loss
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        #
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()



    def pickOne(self, state):  # Do a random action
        a = Action().sample()
        if a == Action("Migrate"):
            neighs = state["feat"][:, 0][state["feat"][:, 0] >= 0]
            assert len(neighs) > 0
            a.dst = np.random.choice(neighs[1:], 1)
            a.relative_dst = np.where(neighs==a.dst[0])[0]
            assert len(a.dst)==len(a.relative_dst)
        return a
