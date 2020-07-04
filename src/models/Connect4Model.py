# import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import random
# from IPython.display import display

import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import count
from tqdm import tqdm

# =====================================================================================
# =====================================================================================
# ==================================== TRAIN ========================================
# =====================================================================================
# =====================================================================================

class connect_x:

    def __init__(self):
        self.board_height = 6
        self.board_width = 7
        self.board_state = np.full((self.board_height, self.board_width), 0, dtype=np.int8)
        self.players = {'p1': 1, 'p2': 2}
        self.isDone = False
        self.reward = {'win': 1, 'draw': 0.5, 'lose': -1}
    
    def render(self):
        rendered_board_state = self.board_state.copy().astype(np.str)
        rendered_board_state[self.board_state == 0] = ' '
        rendered_board_state[self.board_state == 1] = 'O'
        rendered_board_state[self.board_state == 2] = 'X'
        # Peep this
        # display(pd.DataFrame(rendered_board_state))
    
    def reset(self):
        self.__init__()
        
    def get_available_actions(self):
        available_cols = []
        for j in range(self.board_width):
            # Checking the columns if completely filled to top
            if np.sum([self.board_state[:, j] == 0]) != 0:
                available_cols.append(j)
        return available_cols
    
    def check_game_done(self, player):
        if player == 'p1':
            check = '1 1 1 1'
        else:
            check = '2 2 2 2'
        
        # check vertically then horizontally
        for j in range(self.board_width):
            if check in np.array_str(self.board_state[:, j]):
                self.isDone = True
        for i in range(self.board_height):
            if check in np.array_str(self.board_state[i, :]):
                self.isDone = True
        
        # check left diagonal and right diagonal
        for k in range(0, self.board_height - 4 + 1):
            left_diagonal = np.array([self.board_state[k + d, d] for d in \
                            range(min(self.board_height - k, min(self.board_height, self.board_width)))])
            right_diagonal = np.array([self.board_state[d + k, self.board_width - d - 1] for d in \
                            range(min(self.board_height - k, min(self.board_height, self.board_width)))])
            if check in np.array_str(left_diagonal) or check in np.array_str(right_diagonal):
                self.isDone = True
        for k in range(1, self.board_width - 4 + 1):
            left_diagonal = np.array([self.board_state[d, d + k] for d in \
                            range(min(self.board_width - k, min(self.board_height, self.board_width)))])
            right_diagonal = np.array([self.board_state[d, self.board_width - 1 - k - d] for d in \
                            range(min(self.board_width - k, min(self.board_height, self.board_width)))])
            if check in np.array_str(left_diagonal) or check in np.array_str(right_diagonal):
                self.isDone = True
        
        if self.isDone:
            return self.reward['win']
        # check for draw
        # If every position is taken its draw
        elif np.sum([self.board_state == 0]) == 0:
            self.isDone = True
            return self.reward['draw']
        else:
            return 0.
        
    def make_move(self, a, player):
        # check if move is valid
        if a in self.get_available_actions():
            # calculates lowest empty row
            i = np.sum([self.board_state[:, a] == 0]) - 1
            self.board_state[i, a] = self.players[player]
        else:
            print('Move is invalid')
            self.render()

        reward = self.check_game_done(player)
        
        # give feedback as new state and reward
        return self.board_state.copy(), reward

env = connect_x()

import random

# memory block for deep q learning
class replayMemory:
    def __init__(self):
        self.memory = []
        
    def dump(self, transition_tuple):
        self.memory.append(transition_tuple)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
memory = replayMemory()

class DQN(nn.Module):
    
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.outputs = outputs
        # 6 by 7, 10 by 11 
        # THESE are the different layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(32, 32, kernel_size=5, padding=2)

        linear_input_size = 6 * 7 * 32
        # THESE nn.Linear are just setting up our layers (MP1 = input, MP2 & MP3 = hidden, MP4 = output)
        self.MLP1 = nn.Linear(linear_input_size, 50)
        self.MLP2 = nn.Linear(50, 50)
        self.MLP3 = nn.Linear(50, 50)
        self.MLP4 = nn.Linear(50, outputs)

    # BEING a feed-forward neural network, "forward" sets the steps that the nn should follow
    def forward(self, x):
        # FUNFACT: "relu" stands for rectified linear
        # THIS leaky_relu is performing the activation function on each layer
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        # flatten the feature vector except batch dimension
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.MLP1(x))
        x = F.leaky_relu(self.MLP2(x))
        x = F.leaky_relu(self.MLP3(x))
        # HERE notice how there is no activation function on the output
        return self.MLP4(x)

# ====== SETTING THE DEVICE STATE AS CPU OR GPU (needed info for computations) =========
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# epsilon decay graph
# controls somewhat learn rate or no?
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000

steps_done = np.arange(20000)
eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1 * steps_done / EPS_DECAY)

import torch.optim as optim
import math

BATCH_SIZE = 256
GAMMA = 0.999 # ==================

# get max no. of actions from action space
n_actions = env.board_width

height = env.board_height
width = env.board_width

policy_net = DQN(n_actions).to(device)
# target_net will be updated every n episodes to tell policy_net a better estimate of how far off from convergence
target_net = DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
# set target_net in testing mode
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())

def select_action(state, available_actions, steps_done=None, training=True):
    # batch and color channel
    state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(dim=0).unsqueeze(dim=0) 
    epsilon = random.random() # 0-1
    if training:
        # print(EPS_END, EPS_START, EPS_DECAY, steps_done)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * 1 / EPS_DECAY) # 0.9-0.05
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY) # ================== ? =========================================
    else:
        eps_threshold = 0
    
    # follow epsilon-greedy policy
    if epsilon > eps_threshold:
        with torch.no_grad(): # ================== ? =========================================
            # action recommendations from policy net
            # grabs policy_net's (which is a 2d array) 1st row
            r_actions = policy_net(state)[0, :]
            # print("THESE ARE R ACTIONS", r_actions)
            # separates each tensor/scalar into its own index in array
            state_action_values = [r_actions[action] for action in available_actions]
            # states which index has highest value
            argmax_action = np.argmax(state_action_values)
            # interprets highest value index AKA decision maker
            greedy_action = available_actions[argmax_action]
            return greedy_action
    else:
        return random.choice(available_actions)

def optimize_model():
    # if theres enough memory for this computation
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    state_batch, action_batch, reward_batch, next_state_batch = zip(*[(np.expand_dims(m[0], axis=0), \
                                        [m[1]], m[2], np.expand_dims(m[3], axis=0)) for m in transitions])
    # print("THESE ARE TRANSITIONS", [m for m in transitions])
    # tensor wrapper
    state_batch = torch.tensor(state_batch, dtype=torch.float, device=device)
    action_batch = torch.tensor(action_batch, dtype=torch.long, device=device)
    reward_batch = torch.tensor(reward_batch, dtype=torch.float, device=device)
    
    # for assigning terminal state value = 0 later
    # =========================== || =============================
    non_final_mask = torch.tensor(tuple(map(lambda s_: s_[0] is not None, next_state_batch)), device=device)
    non_final_next_state = torch.cat([torch.tensor(s_, dtype=torch.float, device=device).unsqueeze(0) for s_ in next_state_batch if s_[0] is not None])
    
    # prediction from policy_net
    # ============================ || =============================
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # truth from target_net, initialize with zeros since terminal state value = 0
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # tensor.detach() creates a tensor that shares storage with tensor that does not require grad
    next_state_values[non_final_mask] = target_net(non_final_next_state).max(1)[0].detach()
    # compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1)) # torch.tensor.unsqueeze returns a copy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def random_agent(actions):
    return random.choice(actions)

# win rate test
def win_rate_test():
    win_moves_taken_list = []
    win = []
    for i in range(100):
        env.reset()
        win_moves_taken = 0

        while not env.isDone:
            state = env.board_state.copy()
            available_actions = env.get_available_actions()
            action = select_action(state, available_actions, training=False)
            state, reward = env.make_move(action, 'p1')
            win_moves_taken += 1

            if reward == 1:
                win_moves_taken_list.append(win_moves_taken)
                win.append(1)
                break

            available_actions = env.get_available_actions()
            action = random_agent(available_actions)
            state, reward = env.make_move(action, 'p2')

    return sum(win)/100, sum(win_moves_taken_list)/len(win_moves_taken_list)

steps_done = 0
training_history = []

def train():
    num_episodes = 20000
    # control how lagged is target network by updating every n episodes
    TARGET_UPDATE = 10

    for i in tqdm(range(num_episodes)): 
        env.reset()
        state_p1 = env.board_state.copy()

        # record every 20 epochs
        if i % 20 == 19:
            win_rate, moves_taken = win_rate_test()
            training_history.append([i + 1, win_rate, moves_taken])
            th = np.array(training_history)
            # print training message every 200 epochs
            if i % 200 == 199:
                print('Episode {}: | win_rate: {} | moves_taken: {}'.format(i, th[-1, 1], th[-1, 2]))

        for t in count():
            available_actions = env.get_available_actions()
            action_p1 = select_action(state_p1, available_actions, steps_done)
            steps_done += 1
            state_p1_, reward_p1 = env.make_move(action_p1, 'p1')
            
            if env.isDone:
                if reward_p1 == 1:
                    # reward p1 for p1's win
                    memory.dump([state_p1, action_p1, 1, None])
                else:
                    # state action value tuple for a draw
                    memory.dump([state_p1, action_p1, 0.5, None])
                break
            
            available_actions = env.get_available_actions()
            action_p2 = random_agent(available_actions)
            state_p2_, reward_p2 = env.make_move(action_p2, 'p2')
            
            if env.isDone:
                if reward_p2 == 1:
                    # punish p1 for (random agent) p2's win 
                    memory.dump([state_p1, action_p1, -1, None])
                else:
                    # state action value tuple for a draw
                    memory.dump([state_p1, action_p1, 0.5, None])
                break
            
            # punish for taking too long to win
            memory.dump([state_p1, action_p1, -0.05, state_p2_])
            state_p1 = state_p2_
            
            # Perform one step of the optimization (on the policy network)
            optimize_model()
            
        # update the target network, copying all weights and biases in DQN
        if i % TARGET_UPDATE == TARGET_UPDATE - 1:
            target_net.load_state_dict(policy_net.state_dict())

        if i == 19999:
            # torch.save(target_net.state_dict(), f"target_episode_{i}.pth")
            torch.save(policy_net.state_dict(), f"model_collection/policy_episode.pth")

    print('Complete')

# =====================================================================================
# =====================================================================================
# ==================================== PREDICT ========================================
# =====================================================================================
# =====================================================================================

def connect4Predict(state, available_actions, steps_done=None, training=True):
    old_policy_net = torch.load(os.path.join(os.path.dirname(__file__), 'model_collection/policy_episode.pth'), map_location=torch.device('cpu'))
    policy_net.load_state_dict(old_policy_net)
    # batch and color channel
    state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(dim=0).unsqueeze(dim=0) 
    epsilon = random.random() # 0-1
    if training:
        # print(EPS_END, EPS_START, EPS_DECAY, steps_done)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * 1 / EPS_DECAY) # 0.9-0.05
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY) # ================== ? =========================================
    else:
        eps_threshold = 0
    
    # follow epsilon-greedy policy
    if epsilon > eps_threshold:
        with torch.no_grad(): # ================== ? =========================================
            # action recommendations from policy net
            # grabs policy_net's (which is a 2d array) 1st row
            r_actions = policy_net(state)[0, :]
            # print("THESE ARE R ACTIONS", r_actions)
            # separates each tensor/scalar into its own index in array
            state_action_values = [r_actions[action] for action in available_actions]
            # states which index has highest value
            argmax_action = np.argmax(state_action_values)
            # interprets highest value index AKA decision maker
            greedy_action = available_actions[argmax_action]
            return greedy_action
    else:
        return random.choice(available_actions)
