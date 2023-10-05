# https://blog.paperspace.com/creating-custom-environments-openai-gym/

import gym

from rl import OpenAIGymAdapter

class TrainApplication():
    def __init__(self):
        self._env = OpenAIGymAdapter()

    def run(self):
        obs = self._env.reset()

        while True:
            # Take a random action
            action = self._env.action_space.sample()
            obs, reward, done, info = self._env.step(action)

            # Render the game
            self._env.render()

            if done == True:
                break

        self._env.close()

if __name__ == "__main__":
    pass
