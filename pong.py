# Jared Williams
# Policy gradient for the game of pong

import gym
import numpy as np
import tensorflow as tf

def main():
    """ runs the actions of the program"""

    env = gym.make("Pong-v0")

    for episode in range(100):
        env.reset()
        while True:
            env.render()

            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(obs)


def preprocess_data():
    img = obs[1:176:2, ::2]

def create_model():
    """ creates the arcitecture of the model """

    model = tf.keras.models.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3),
                            activation='relu',
                            input_shape=(210, 160, 1)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    main()

