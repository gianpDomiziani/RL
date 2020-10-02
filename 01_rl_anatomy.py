import random


class Environment():

    def __init__(self):
        self.steps_left = 100

    def get_observations(self) -> [float]:
        return [0.0, 0.0, 0.0]

    def get_actions(self) -> [int]:
        return [0, 1, 2]

    def is_done(self) -> bool:
        return self.steps_left == 0

    def action(self, action: int) -> float:
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        return random.random()

class Agent():

    def __init__(self):
        self.total_reward = 0.0

    def step(self, env: Environment):
        current_obs = env.get_observations()
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward


if __name__ == '__main__':
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)

    print(f"Total reward got: {agent.total_reward}")