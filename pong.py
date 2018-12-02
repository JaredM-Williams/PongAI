# Jared Williams
# Policy gradient for the game of pong

import gym
import numpy as np
import tensorflow as tf

def main():
    env = gym.make("Pong-v0")

    for episode in range(100):
        env.reset()
        while True:
            env.render()

            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(obs)


if __name__ == '__main__':
    main()

