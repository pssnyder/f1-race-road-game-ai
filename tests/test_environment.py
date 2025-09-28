import os
import sys

# Ensure project root is in path
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.environment import F1RaceEnvironment


def test_env_reset_and_step_headless():
    env = F1RaceEnvironment(render=False)
    s = env.reset()
    assert s.shape == (5,)
    for _ in range(5):
        s2, r, done, info = env.step(0)
        assert s2.shape == (5,)
        assert isinstance(r, float)
        assert isinstance(done, bool)
        assert "score" in info
    env.close()
