import random
from collections import deque, namedtuple

import numpy as np
import torch
from numpy.random import RandomState
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

from action import Action, action_sample
from models import MLP

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', "done"))

class Agent():

    def __init__(self, num_node_features, num_classes):

        #TODO ADD HYPERPARAMETERS as ARGS of __INIT__
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.05  # exploration rate
        self.rnd = RandomState(0)  # seed
        self.batch_size = 3
        self.tau = 2e-3
        self.learn_rate = 0.01
        self.memory = deque(maxlen=50) # bigger than batch_size. fix by experimentations

        ####

        self.model = MLP(num_node_features=num_node_features, hidden_channels=8, num_classes=num_classes)
        self.target_model = MLP(num_node_features=num_node_features, hidden_channels=8, num_classes=num_classes)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)
        self.loss_op = torch.nn.CrossEntropyLoss()

    def act(self, state,action_space):
        v = self.rnd.rand()
        if v <= self.epsilon:
            return self.pickOne(state,action_space)
        else:
            node_feats = torch.tensor(state.x, dtype=torch.float)
            edge_index = torch.tensor(state.edge_index, dtype=torch.long)
            data = Data(x=node_feats, edge_index=edge_index)
            self.model.eval()
            with torch.no_grad():
                action_values = self.model(data.x, data.edge_index)
            self.model.train()

            # print(data)
            # print("A",action_values)

            # type_oper = int(np.argmax(action_values)) # max. value of all probabilities, the index is the class of the operation
            type_oper = int(action_values.max(1).indices[0])
            # print(action_values.shape)
            pos_relative_action = int(np.argmax(action_values.max(1)[0])) # which node
            #TODO - Pointless Operations - "NoOperation in other node" are possible. Thus,  Can they be penalised?

            # print("type_oper",type_oper)
            # print("pos_relative_action",pos_relative_action)
            a = Action(action_space,type_oper,pos_relative_action)

            # print(a)
            return a


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_loader(self, samples):
        data_list = []
        for state in samples:
            node_feats = torch.tensor(state.x, dtype=torch.float)
            edge_index = torch.tensor(state.edge_index, dtype=torch.long)
            data = Data(x=node_feats, edge_index=edge_index)
            data_list.append(data)
        loader = DataLoader(data_list, batch_size=self.batch_size)
        return loader

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        # https://deeplearning.neuromatch.io/tutorials/W3D2_BasicReinforcementLearning/student/W3D2_Tutorial1.html#coding-exercise-6-1-implement-nfq

        samples = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*samples)) #bestial

        batch_actions = np.array([act.space for act in batch.action]) # get the action.space from Action type
        batch_actions = torch.tensor(batch_actions, dtype=torch.long)
        # print(batch.state)
        # print(batch_actions)
        # Q on S,a
        loader_states = self.get_loader(batch.state)
        data = list(loader_states)[0]  # loader len equals batch_size

        model_batch = self.model(data.x, data.edge_index, data.batch)
#test
        # Qsa = model_batch.gather(1,batch_actions).unsqueeze(-1)
        Qsa = model_batch.gather(1,batch_actions.view(-1,1)).squeeze(0)
        # print("B) QSA ")
        # print(Qsa)


        # Q on s', a'
        loader_states_prime = self.get_loader(batch.next_state)
        data_next_state = list(loader_states_prime)[0]  # loader len equals batch_size


        # Compute Q targets for current states
        rewards = torch.tensor(batch.reward, dtype=torch.long) # TODO create torch(Rewards) in env.?
        # rewards = rewards.reshape(12,1)
        # print("Rewards")
        # print(rewards)
        dones = ByteTensor(batch.done)
        # print("done")
        # print(dones)

        with torch.no_grad():
            Qsa_prime_target_values = self.target_model(data_next_state.x,data_next_state.edge_index,data_next_state.batch).detach()
            # print(Qsa_prime_target_values)
            # print(Qsa_prime_target_values.shape)

            # Qsa_prime_targets = Qsa_prime_target_values.max(axis=-1)[0]
            Qsa_prime_targets = Qsa_prime_target_values.max(1)[0].unsqueeze(1)

            # print("Qsa_prime_targets")
            # print(Qsa_prime_targets)

            Qsa_targets = rewards + (self.gamma * Qsa_prime_targets ) #* (1 - dones))
            # print("A) Qsa_targets ")
            Qsa_targets = Qsa_targets.squeeze(1)
            # print(Qsa_targets)



        # Compute loss (error)
        loss = F.mse_loss(Qsa_targets, Qsa)
        # print("LOSS: ",loss)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.model, self.target_model, self.tau)
        return None


        # non_final_mask = ByteTensor(batch.done).neg() # Non final states are ok, so done==False is True,
        #
        # #non_final_next_states = Variable(torch.cat([s for (s,d) in zip(batch.next_state,batch.done) if d is False]),volatile=True)
        #
        # non_final_next_states = [s for (s,d) in zip(batch.next_state,batch.done) if d is False]
        #
        # print(len(samples))
        # print(len(non_final_mask))
        # print(len(non_final_next_states))
        #
        # # return None
        # # non_final_next_states = list(self.__get_loader_states(non_final_next_states))[0]
        #
        # # print(non_final_next_states)
        # #El problema es que el non_final_maks debería de tener 4 (salidas) por cada done
        # # non_final_mask = [int(done) for *rest, done in samples ] #could be improved... * *features
        # # non_final_mask = ByteTensor(non_final_mask)
        #
        # loader_states = self.get_loader(batch.state)
        # data = list(loader_states)[0] #loader len equals batch_size
        #
        # print("T ",[float(a.code) for a in batch.action])
        #
        # action_batch = Variable(torch.cat([FloatTensor(a.code) for a in batch.action]))
        # print("AB:",action_batch)
        #
        # model_batch = self.model(data.x, data.edge_index,data.batch)
        #
        # state_action_values = model_batch.gather(1, action_batch)
        #
        # print(state_action_values)
        # return None
        #
        #
        # output, state_action_values = self.model(data.x, data.edge_index,data.batch).max(dim=1)
        #
        # next_state_values = Variable(torch.zeros(self.batch_size).type(Tensor))
        #
        # print(next_state_values)
        # print(non_final_mask)
        # print(self.model(non_final_next_states.x,non_final_next_states.edge_index).max(dim=1)[0])
        # next_state_values[non_final_mask] =  self.model(non_final_next_states.x,non_final_next_states.edge_index).max(dim=1)
        #

        # print(data)
        # print("P:",pred)
        # print("Y:",data.y)
        # print("-" * 10)

        # # Compute the expected Q values
        # reward_batch = Variable(torch.cat(batch.reward))
        # expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        #
        # # Compute Huber loss
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        #
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

    #
    """
    Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    
    FROM: https://github.com/xkiwilabs/DQN-using-PyTorch-and-ML-Agents/blob/8bd47f7c845bbbab2cbb34d717dd08c8b7a50aab/dqn_agent.py#L74
    
    """
    def soft_update(self, local_model, target_model, tau):
        """
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def pickOne(self, state, action_space):  # Do a random action
        a = action_sample(action_space)
        if a == "Migrate":
            neighs = state.x[:, 0][state.x[:, 0] >= 0]
            assert len(neighs) > 0
            a.dst = np.random.choice(neighs[1:], 1)
            a.relative_dst = np.where(neighs==a.dst[0])[0]
            assert len(a.dst)==len(a.relative_dst)
        return a
