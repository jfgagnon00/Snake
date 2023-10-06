from gymnasium.envs.registration import register
from .envs.SnakeEnvironment import SnakeEnvironment

register(
    id="snake/SnakeEnvironment-v0",
    entry_point=SnakeEnvironment
)
