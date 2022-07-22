import numpy as np
from Q_learning import Qlearning
from Agent_Q import AgentQ

if __name__ == '__main__':
    states = np.array([0, 1, 2, 3])
    actions = np.array([0, 1])
    a1 = AgentQ(states, actions, 0)
    a2 = AgentQ(states, actions, 0)
    q_l = Qlearning((a1, a2), learning_rate=0.2, discount_factor=0.95, iter=300000)
    q_l.run_multiple_iterated_games(100)