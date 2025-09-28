import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.agent import DQNAgent


def test_agent_action_and_memory():
    agent = DQNAgent(state_size=5, action_size=3, memory_size=100, batch_size=8)
    state = np.zeros(5, dtype=np.float32)
    action = agent.choose_action(state, training_mode=True)
    assert action in (0, 1, 2)
    agent.store_experience(state, action, 0.0, state, False)
    assert len(agent.memory) == 1
