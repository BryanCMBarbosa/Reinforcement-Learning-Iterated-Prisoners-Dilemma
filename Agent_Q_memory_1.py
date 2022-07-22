import numpy as np
from cmath import nan
from Agent import Agent
import matplotlib.pyplot as plt

class AgentQ_1mem(Agent):
    def __init__(self, states, actions, initial_state:int):
        Agent.__init__(self, states, actions, initial_state)
        self.q_table = np.zeros((len(states), len(actions)))
        self.boltzmann_table = self.boltzmann_table = np.zeros((len(states), len(actions)))
        self.previous_state = 0

    def calculate_boltzmann_matrix(self, temperature):
        if ((self.q_table > 7).any()):
            q_table_exp = self.q_table/temperature #np.exp
        else:
            q_table_exp = np.exp(self.q_table/temperature)

        self.boltzmann_table = q_table_exp/np.sum(q_table_exp, axis=1)[:, None]

    def play(self, temperature):
        if temperature >= 0.01:
            self.calculate_boltzmann_matrix(temperature)
            self.action = np.random.choice(self.actions, p = self.boltzmann_table[self.state, :])
        else:
            self.action = np.argmax(self.q_table[self.state, :])
        
        self.chosen_actions = np.append(self.chosen_actions, self.action)
        self.increment_coop_and_defect()

        #return self.action

    def convert_binary(self, value):
        binary = format(value, "b")
        while len(binary) < 4:
            binary = "0" + binary

        return binary

    def decide_next_state(self, oponent_action):  #0000 - CC+CC - 0 / 0001 - CC+CB - 1 / 0010 - CC+BC - 2 / 0011 - CC+BB - 3 / 0100 - CB+CC - 4 / 0101 - CB+CB - 5 / 0110 - CB+BC - 6 / 0111 - CB+BB - 7 / 1000 - BC+CC - 8 / 1001 - BC+CB - 9 / 1010 - BC+BC - 10 / 1011 - BC+BB - 11 / 1100 - BB+CC - 12 / 1101 - BB+CB - 13 / 1110 - BB+BC - 14 / 1111 - BB+BB - 15
        state_str = self.convert_binary(self.state)
        if state_str[2] == "0" and state_str[3] == "0":
            if self.action == oponent_action:
                if self.action == 0:
                    self.next_state = 0
                else:
                    self.next_state = 3
            else:
                if self.action == 0:
                    self.next_state = 1
                else:
                    self.next_state = 2

        elif state_str[2] == "0" and state_str[3] == "1":
            if self.action == oponent_action:
                if self.action == 0:
                    self.next_state = 4
                else:
                    self.next_state = 7
            else:
                if self.action == 0:
                    self.next_state = 5
                else:
                    self.next_state = 6

        elif state_str[2] == "1" and state_str[3] == "0":
            if self.action == oponent_action:
                if self.action == 0:
                    self.next_state = 8
                else:
                    self.next_state = 11
            else:
                if self.action == 0:
                    self.next_state = 9
                else:
                    self.next_state = 10

        else:
            if self.action == oponent_action:
                if self.action == 0:
                    self.next_state = 12
                else:
                    self.next_state = 15
            else:
                if self.action == 0:
                    self.next_state = 13
                else:
                    self.next_state = 14

    def upgrade_score(self, learning_rate, discount_factor):
        self.q_table[self.state, self.action] = (1 - learning_rate)*self.q_table[self.state, self.action] + learning_rate*(self.rewards[-1] + discount_factor*np.max(self.q_table[self.next_state, :]))

    def upgrade_state(self):
        self.previous_state = self.state
        self.state = self.next_state

    def print_data(self):
        print("\nQ AGENT MEMORY 1 STATISTICS")
        print("Coops: ", self.coops)
        print("Defections: ", self.defects)
        print("Average points per game: ", np.average(self.rewards))
        print(self.q_table)
    
    def plot_data(self):
        plt.plot(self.chosen_actions, linewidth=0.8)
        plt.xlabel("Iteration")
        plt.ylabel("0: Cooperation | 1: Defection")
        plt.show()

    def reset(self):
        self.state = 0
        self.action = nan
        self.coops = 0
        self.defects = 0
        self.chosen_actions = np.array([])
        self.rewards = np.array([])
        self.next_state = nan
        self.q_table = np.zeros((len(self.states), len(self.actions)))
        self.boltzmann_table = self.boltzmann_table = np.array((len(self.states), len(self.actions)))

