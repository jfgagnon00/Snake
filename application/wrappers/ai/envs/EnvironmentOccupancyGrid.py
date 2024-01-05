import gymnasium as gym


class EnvironmentOccupancyGrid(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = env.observation_space["occupancy_grid"]

    def reset(self, *args, seed=None, options=None):
        observations, info = self.env.reset(*args, seed=seed, options=options)
        return observations["occupancy_grid"], info

    def step(self, *args):
        observations, reward, terinated, truncated, info = self.env.step(*args)
        return observations["occupancy_grid"], reward, terinated, truncated, info

    def observation(self, observations):
        return observations["occupancy_grid"]
