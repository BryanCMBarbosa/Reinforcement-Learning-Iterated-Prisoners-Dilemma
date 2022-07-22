from cmath import nan
from matplotlib.pyplot import axis
import numpy as np
import matplotlib.pyplot as plt
import Agent_Q
import Agent_TFT


class Agent_TFT():
    def __init__(self, initial_action = 0):
        self.action = initial_action
        self.rewards = np.array([])
        self.coops = 0
        self.defections = 0

    def set_last_oponent_action(self, last_oponent_action:int):
        self.action = last_oponent_action
    
    def play(self):
        return self.action


class Agent_Q():
    def __init__(self, states, actions, initial_state:int):
        self.states = states
        self.actions = actions
        self.q_table = np.zeros((len(states), len(actions)))
        self.boltzmann_table = np.array((len(states), len(actions)))
        self.state = initial_state
        self.action = nan
        self.rewards = np.array([])
        self.chosen_actions = np.array([])
        self.coops = 0
        self.defections = 0

    def q_new_state(self, tft_action): #00 - CC - 0 / 01 - CB - 1 / 10 - BC - 2 / 11 - BB - 3 
        if self.action == tft_action:
            if self.action == 0:
                return 0
            else:
                return 3
        else:
            if self.action == 0:
                return 1
            else:
                return 2
    
    def add_reward(self, reward):
        self.rewards = np.append(self.rewards, reward)

    def upgrade_q_table(self, learning_rate, discount_factor, new_state):
        self.q_table[self.state, self.action] = (1 - learning_rate)*self.q_table[self.state, self.action] + learning_rate*(self.rewards[-1] + discount_factor*np.max(self.q_table[new_state, :]))

    def calculate_boltzmann_matrix(self, temperature):
        q_table_exp = np.exp(self.q_table/temperature)
        self.boltzmann_table = q_table_exp/np.sum(q_table_exp, axis=1)[:, None]

class Q_learning():
    def __init__(self, agents, agentQ, agenttft, learning_rate = 0.1, discount_factor = 0.99, exploration_rate = 1, max_exploration_rate = 1, min_exploration_rate = 0.01, exploration_decay_rate = 0.0001):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.agents = agents
        self.agentQ = agentQ
        self.agenttft = agenttft
        self.num_episodes = 0
        self.temperature = 5.0
        self.skip_rows = []

    def give_rewards(self, actions):
        if actions[0] == actions[1]:
            if actions[0] == 0:
                return 0.3, 0.3
            else:
                return 0.1, 0.1
        else:
            if actions[0] == 1:
                return 0.5, 0.0
            else:
                return 0.0, 0.5

    def calculate_temperature(self, n):
        return 5*(0.999**n)
    
    def increment_coop_and_defect(self):
        if self.agentQ.action == 0:
            self.agentQ.coops+=1
        else:
            self.agentQ.defections+=1

        if self.agenttft.action == 0:
            self.agenttft.coops+=1
        else:
            self.agenttft.defections+=1


    def run_episode(self, episode_num):
        """"
        exploitation = random.uniform(0.0, 1.0)
        if exploitation > self.exploration_rate:
            self.agentQ.action = np.argmax(self.agentQ.q_table[self.agentQ.state, :])
        else:
            self.agentQ.action = np.random.choice(self.agentQ.actions)
        
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate -= self.exploration_decay_rate   
        """
        if self.temperature >= 0.3:
            self.temperature = self.calculate_temperature(episode_num)
            
            b_matrix = self.agentQ.calculate_boltzmann_matrix(self.agentQ.q_table, self.temperature)
            self.agentQ.action = np.random.choice(self.agentQ.actions, p = b_matrix[self.agentQ.state, :])
        else:
            self.agentQ.action = np.argmax(self.agentQ.q_table[self.agentQ.state, :])

        q_new_state = self.agentQ.q_new_state(self.agenttft.action)
        self.agentQ.chosen_actions = np.append(self.agentQ.chosen_actions, self.agentQ.action)
        
        reward_q, reward_tft = self.give_rewards((self.agentQ.action, self.agenttft.action))
        self.agentQ.rewards = np.append(self.agentQ.rewards, reward_q)
        self.agenttft.rewards = np.append(self.agenttft.rewards, reward_tft)

        self.agentQ.q_table[self.agentQ.state, self.agentQ.action] = (1 - self.learning_rate)*self.agentQ.q_table[self.agentQ.state, self.agentQ.action] + self.learning_rate*(reward_q + self.discount_factor*np.max(self.agentQ.q_table[q_new_state, :]))

        self.increment_coop_and_defect()

        self.agentQ.state = q_new_state
        self.agenttft.set_last_oponent_action(self.agentQ.action)


    def train(self, iterations):
        for i in range(iterations):
            self.run_episode(i)
            self.num_episodes+=1

    def print_data(self):
        print("Q AGENT STATISTICS")
        print("Coops: ", self.agentQ.coops)
        print("Defections: ", self.agentQ.defections)
        print("Average points per game: ", np.average(self.agentQ.rewards))
        print("Learned strategy for CC: ", np.argmax(self.agentQ.q_table[0, :]))
        print("Learned strategy for CB: ", np.argmax(self.agentQ.q_table[1, :]))
        print("Learned strategy for BC: ", np.argmax(self.agentQ.q_table[2, :]))
        print("Learned strategy for BB: ", np.argmax(self.agentQ.q_table[3, :]))
        print(self.agentQ.q_table)
        plt.plot(self.agentQ.chosen_actions, linewidth=0.8)
        plt.xlabel("Iteration")
        plt.ylabel("0: Cooperation | 1: Defection")
        plt.show()
        print("\n\n\nTFT AGENT STATISTICS")
        print("Coops: ", self.agenttft.coops)
        print("Defections: ", self.agenttft.defections)
        print("Average points per game: ", np.average(self.agentQ.rewards))
        
            
if __name__ == '__main__':
    states = np.array([0, 1, 2, 3])
    actions = np.array([0, 1])
    a_q = Agent_Q(states, actions, 0)
    a_tft = Agent_TFT(0)
    q_l = Q_learning(a_q, a_tft, learning_rate=1.0, discount_factor=0.05)
    q_l.train(100000)
    q_l.print_data()