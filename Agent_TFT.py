import numpy as np
from cmath import nan
from Agent import Agent

class AgentTFT(Agent):
    def __init__(self, states, actions, initial_state:int):
        Agent.__init__(self, states, actions, initial_state)

    def decide_next_state(self, oponent_action):
        self.next_state = oponent_action
    
    def upgrade_state(self):
        self.state = self.next_state
        
    def upgrade_score(self, learning_rate, discount_factor):
        pass

    def play(self, temperature):
        self.action = self.state
        self.increment_coop_and_defect()
        self.chosen_actions = np.append(self.chosen_actions, self.action)

    def print_data(self):
        print("\nTFT AGENT STATISTICS")
        print("Coops: ", self.coops)
        print("Defections: ", self.defects)
        print("Average points per game: ", np.average(self.rewards))
        
    def reset(self):
        self.state = 0
        self.action = nan
        self.coops = 0
        self.defects = 0
        self.chosen_actions = np.array([])
        self.rewards = np.array([])
        self.next_state = nan