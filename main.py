import random
from timeit import default_timer as timer

import networkx as nx
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from DQNAgent import Agent
from action import Action, action_sample, ACTION_NAMES
from gymFogEnv import FogEnv

seed = 0
random.seed = seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging_dir = "tmp/tensorlogs/"
writer = SummaryWriter(logging_dir)

# Topology
G = nx.balanced_tree(2, 5)
edge_nodes = [x for (x, d) in nx.degree(G) if d == 1]
pos = nx.spring_layout(G, seed=seed)

env = FogEnv(G, pos, edge_nodes)
env.seed(seed)
state, done = env.reset()

# env.render(text='episode: XX, action: XX, reward: XX',path="tmp/images/image0.png")
# Image(filename="tmp/images/image0.png")

# Environment from topology
env = FogEnv(G, pos, edge_nodes)
env.seed(seed)
env.reset()
print("Max. action space ", env.max_degree)

episodes = 3
frame = 1
render = True
batch_size = 10

agent = Agent(num_node_features=env.num_features, num_classes=len(ACTION_NAMES))

if render:
    env.render(text='episode: 0, step: 0, action: -, reward: -',
               path="tmp/images/image_%08d.png" % (0))

# for episode in tqdm(range(episodes)):
for episode in range(1,episodes+1):
    # run episode
    start_time = timer()
    start_frame = frame

    # initialize the episode
    state, done = env.reset()

    score = 0
    while not done:

        action = agent.act(state,env.action_space)
        # action = agent.pickOne(state,env.action_space)
        # print(state["feat"])
        # print("HERE")
        print(action)

        state_next, reward, done = env.step(action)

        agent.remember(state, action, reward, state_next, done)
        loss = agent.optimize_model()  # or train the model after each action?
        #
        # if render:
        #     env.render(text='episode: {}, step: {}, action: {}, reward: {}'.format(episode, frame - start_frame, action,
        #                                                                            reward),
        #                path="tmp/images/image_%08d.png" % (frame))
        #
        state = state_next
        score += reward
        frame += 1


    # end_time = timer()
    # fps = (frame - start_frame) / (end_time - start_time)
    # # To comment
    # print('episode: {}, frame: {}, fps: {}, score: {}'.format(episode, frame, int(fps), score))
    #
    # writer.add_scalar('fps', fps, episode)
    # writer.add_scalar('score/frame', score, episode)
    # writer.add_scalar('score/episode', score, episode)

writer.close()
