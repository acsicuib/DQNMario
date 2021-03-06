import gym
import networkx as nx
import numpy as np
from action import Action
import matplotlib.pyplot as plt
import PIL
from gym import spaces
from collections import namedtuple

State = namedtuple("State",
                   ("x","edge_index"))

class FogEnv(gym.Env):
    def __init__(self, graph, pos, edge_nodes, id_node_cloud=0):
        super(FogEnv, self).__init__()
        self.G = graph
        self.node_cloud = id_node_cloud
        self.pos = pos
        self.edge_nodes = edge_nodes

        self.max_degree = max(dict(nx.degree(self.G)).values())
        self.action_space = self.max_degree + 1

        # Future issue: level in other type of topologies?
        self.level_node = nx.single_source_shortest_path_length(self.G,
                                                                source=self.node_cloud)  # get tier-level based on Cloud-node length
        self.number_levels = max(self.level_node.values()) + 1
        self.max_steps = nx.diameter(self.G)

        # Load of Graph and Node features. It should be an external process.
        centrality = nx.eigenvector_centrality(self.G)
        features = []
        for n in self.G.nodes():
            hwr = (self.number_levels - self.level_node[n]) * 10
            watts = (0.2 if n % 2 == 0 else 2.4)
            features.append([n, centrality[n], self.level_node[n], hwr, watts, int(n in self.edge_nodes)])

        self.features = np.array(features)
        self.num_features = self.features.shape[1]



    def __get_neighs(self):
        return np.array([n for n in self.G.neighbors(self.agent_alloc)])

    def __do_state(self):
        # print("Agent Alloc is: ",self.agent_alloc)
        neighs = self.__get_neighs()
        diff = max(0,self.action_space - len(neighs) - 1) # we include the current node of the service in 0.position
        node_feats = np.array(self.features[[self.agent_alloc, *neighs]]).astype(float)
        diff_row = (np.ones(self.num_features * diff) * -1).reshape(diff, self.num_features)
        node_feats = np.vstack((node_feats, diff_row))

        edge_index = [[*np.zeros(len(neighs), dtype=int), *np.arange(1, len(neighs) + 1), *[-1] * diff * 2],
                      [*np.arange(1, len(neighs) + 1), *np.zeros(len(neighs), dtype=int), *[-1] * diff * 2]]
        return State(node_feats,edge_index)

    def reset(self):
        self.agent_alloc = self.node_cloud
        self.current_steps = self.max_steps
        return self.__do_state(), False

    def step(self, action):
        reward = 0
        semantic_operation_ok = True
        i_neigh = 0

        if action == "NoOperation":
            None

        if action == "Migrate":
            # print(self.agent_alloc)
            neighs = self.__get_neighs()
            neighs = [self.agent_alloc,*neighs] # the state is composed by [itself node , + neighs]
            # print(neighs)
            assert len(action.relative_dst) > 0
            i_neigh = action.relative_dst[0]

            # assert i_neigh < len(neighs),"Try to reach a fake node " #[-1,-1,-1,...]
            if i_neigh < len(neighs):
                action.dst = []
                action.dst.append(neighs[i_neigh])
            else:
                semantic_operation_ok = False

            # TODO UPDATE  node features -> h, watts, ...

            # Reward
            goal_node = sorted(neighs)[1]  # The second lowest!
            levelpast = self.level_node[self.agent_alloc]

            if not semantic_operation_ok:
                reward -= 100
            elif action.dst[0] == goal_node:
                reward += 10  # Values - Ranges...
            elif action.dst[0] == min(neighs):
                reward -= 5  # go back
            else:
                reward += 2

            # Update structures and var controls
            if semantic_operation_ok:
                self.agent_alloc = int(action.dst[0])

        self.current_steps -= 1

        done = False
        if self.current_steps == 0:
            done = True
            reward = -100
        elif self.agent_alloc in self.edge_nodes:
            done = True
            reward += 10

        # if not done:
        #     reward -= -3
        space_reward = np.zeros(action.size)
        space_reward[i_neigh] = reward

        return self.__do_state(), space_reward, done

    def render(self, **kwargs):
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        # labels = nx.draw_networkx_labels(self.G, self.pos)
        nx.draw(self.G, self.pos, with_labels=False, node_size=100, node_color="#A2A0A0", edge_color="gray",
                node_shape="o", font_size=7, font_color="white", ax=ax)
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=self.edge_nodes, node_color="r", node_size=100, ax=ax)
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[self.node_cloud], node_size=100, node_color="g", ax=ax)
        ax.scatter(self.pos[self.agent_alloc][0], self.pos[self.agent_alloc][1], s=300.0, marker='o', color="orange")

        if "text" in kwargs:
            #             left, bottom, width, height = ax.get_position().bounds
            text = kwargs["text"]
            ax.text(0, 0, text, verticalalignment='top', transform=ax.transAxes, fontsize=16)

        if "path" in kwargs:
            canvas = plt.get_current_fig_manager().canvas
            canvas.draw()
            pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
            pil_image.save(kwargs["path"])

        plt.ioff()
        plt.close()
        return True