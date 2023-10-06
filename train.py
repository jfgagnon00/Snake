"""
Responsable de l'entrainement des agents
"""


import ai
import ai.agents as agents
import gymnasium as gym
import os

from configs import configsCreate


class TrainApplication():
    """
    """
    def __init__(self, configs):
        self._agent = agents.AgentRandom()
        self._env = gym.make("snake/SnakeEnvironment-v0",
                            # render_mode="human",
                            environmentConfig=configs.environment,
                            simulationConfig=configs.simulation,
                            graphicsConfig=configs.graphics)

    def run(self):
        observation = self._env.reset()

        while True:
            # Take a random action
            action = self._agent.getAction(observation)
            state, reward, terminated, truncated, info = self._env.step(action)

            # Render the game
            self._env.render()

            if terminated:
                break

        self._env.close()

if __name__ == "__main__":
    # mettre le repertoire courant comme celui par defaut
    # (facilite la gestion des chemins relatifs)
    path = os.path.abspath(__file__)
    path, _ = os.path.split(path)
    os.chdir(path)

    configs = configsCreate("config_overrides.json")

    TrainApplication(configs).run()
