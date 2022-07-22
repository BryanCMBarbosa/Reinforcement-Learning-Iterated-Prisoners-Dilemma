import numpy as np
import pandas as pd


class Agent_Stats:
    def __init__(self, iter):
        self.iter = iter
        self.coops = np.array([])
        self.defects = np.array([])
        self.chosen_actions = np.empty((0, self.iter), int)
        self.rewards = np.empty((0, self.iter), float)
        self.states = np.empty((0, self.iter), int)

    def turn_chosen_actions_comb_into_states(self, agent2):
        row = np.array([])
        for i, j in zip(self.chosen_actions, agent2.chosen_actions):
            for k, l in zip(i, j):
                if k == l:
                    if k == 0:
                        row = np.append(row, 0)
                    else:
                        row = np.append(row, 3)
                else:
                    if k == 0:
                        row = np.append(row, 1)
                    else:
                        row = np.append(row, 2)
    
            self.states = np.append(self.states, np.array([row]), axis=0)
            row = np.array([])

        return self.states
            


    def turn_into_dataframe(self, option=0):
        if option == 0:
            df = pd.DataFrame(data = self.states[0:,0:], index=[i for i in range(self.states.shape[0])], columns=['Round '+str(i) for i in range(self.states.shape[1])])
        elif option == 1:
            df = pd.DataFrame(data = self.chosen_actions[0:,0:], index=[i for i in range(self.chosen_actions.shape[0])], columns=['Round '+str(i) for i in range(self.chosen_actions.shape[1])])
        elif option == 2:
            df = pd.DataFrame(data = self.rewards[0:,0:], index=[i for i in range(self.rewards.shape[0])], columns=['Round '+str(i) for i in range(self.rewards.shape[1])])

        return df