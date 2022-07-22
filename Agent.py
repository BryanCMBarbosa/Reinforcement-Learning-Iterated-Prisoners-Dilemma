import numpy as np
from cmath import nan

class Agent():
    def __init__(self, states, actions, initial_state:int):
        self.states = states
        self.actions = actions
        self.state = initial_state
        self.action = nan
        self.coops = 0
        self.defects = 0
        self.chosen_actions = np.array([])
        self.rewards = np.array([])
        self.next_state = nan

    def upgrade_state(self):
        pass

    def upgrade_score(self, learning_rate, discount_factor):
        pass

    def play(self, temperature):
        pass

    def increment_coop_and_defect(self):
        if self.action == 0:
            self.coops += 1
        else:
            self.defects += 1
    
    def get_reward(self, reward):
        self.rewards = np.append(self.rewards, reward)



        