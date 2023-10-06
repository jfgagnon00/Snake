from gym.envs.registration import register
from .SnakeEnvironment import SnakeEnvironment

register(
    id="snake/SnakeEnvironment-v0",
    entry_point=SnakeEnvironment
)
