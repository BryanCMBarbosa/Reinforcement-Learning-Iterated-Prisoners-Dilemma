import numpy as np
import matplotlib.pyplot as plt
import Agent
from Agent_Stats import Agent_Stats

class Qlearning():
    def __init__(self, agents, learning_rate = 0.1, discount_factor = 0.95, temperature = 5.0, exploring_annealing_factor = 0.999, iter = 100000):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.agents = agents
        self.num_episodes = 0
        self.initial_temperature = temperature
        self.temperature = temperature
        self.exploring_annealing_factor = exploring_annealing_factor
        self.iter = iter
        self.agents_stats = [Agent_Stats(iter), Agent_Stats(iter)]

    def give_rewards(self, action_0, action_1):
        if action_0 == action_1:
            if action_0 == 0:
                return 0.3, 0.3
            else:
                return 0.1, 0.1
        else:
            if action_0 == 1:
                return 0.5, 0.0
            else:
                return 0.0, 0.5

    def calculate_temperature(self, n):
        if self.temperature >= 0.01:
            self.temperature = 5*(self.exploring_annealing_factor**n)
            return self.temperature
        else:
            return self.temperature

    def run_episode(self, episode_num):
        self.temperature = self.calculate_temperature(episode_num)
        for a in self.agents:
            a.play(self.temperature)

        reward_0, reward_1 = self.give_rewards(self.agents[0].action, self.agents[1].action)
        self.agents[0].get_reward(reward_0)
        self.agents[1].get_reward(reward_1)

        self.agents[0].decide_next_state(self.agents[1].action)
        self.agents[1].decide_next_state(self.agents[0].action)

        for a in self.agents:
            a.upgrade_score(self.learning_rate, self.discount_factor)
            a.upgrade_state()

    def train(self):
        for i in range(self.iter):
            self.run_episode(i)
            self.num_episodes+=1

    def print_data(self):
        for a in self.agents:
            a.print_data()

    def copy_agent_data(self, agent, stat_obj):
        stat_obj.coops =  np.append(stat_obj.coops, agent.coops)
        stat_obj.defects = np.append(stat_obj, agent.defects)
        stat_obj.chosen_actions = np.append(stat_obj.chosen_actions, np.array([agent.chosen_actions]), axis=0)
        stat_obj.rewards = np.append(stat_obj.rewards, np.array([agent.rewards]), axis=0)
        
        return stat_obj

    def append_agents_data(self):
        for a, s in zip(self.agents, self.agents_stats):
            s = self.copy_agent_data(a, s)

    def reset_agents(self):
        for a in self.agents:
            a.reset()

    def reset_game(self):
        self.temperature = self.initial_temperature
        self.num_episodes = 0

    def run_multiple_iterated_games(self, number_of_games):
        for i in range(number_of_games):
            print("Running game", i+1, "of", number_of_games)
            self.train()
            self.append_agents_data()
            self.reset_agents()
            self.reset_game()
        print("Done!")
        self.agents[0].print_data()

