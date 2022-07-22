import numpy as np
from Q_learning import Qlearning
from Agent_Q import AgentQ
from Agent_Q_memory_1 import AgentQ_1mem

if __name__ == '__main__':
    states = np.array([0, 1, 2, 3])
    actions = np.array([0, 1])
    a1 = AgentQ(states, actions, 0)
    a2 = AgentQ_1mem(states, actions, 0)
    q_l = Qlearning((a1, a2), learning_rate = 1.0, discount_factor = 0.05)
    q_l.train(100000)
    q_l.print_data()