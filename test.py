import matplotlib.pyplot as plt
import time
from PIL import Image
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
# import crafter
from crafter import crafter
from utils import env_wrapper
import os
import numpy as np
from tqdm import tqdm
from temp_result import submodel_wrappers1


def test(env, model, num_episodes, stack_size=2, render=True):

    print("Testing...")

    total_rewards = []

    for episode in tqdm(range(num_episodes)):

        obs = env.reset()

        if render:

            plt.ion()
            fig, ax = plt.subplots()
            image_display = ax.imshow(obs)
            plt.show(block=False)

        done = False
        episode_reward = 0
        frames = [obs] * stack_size 

        while not done:

            action, _ = model.predict(np.concatenate(frames, axis=-1), deterministic=False)

            frames.pop(0)
            frames.append(obs)

            obs, reward, done, info = env.step(action)

            episode_reward += reward

            if render:

                img = Image.fromarray(obs)

                image_display.set_data(img)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

                time.sleep(0.2)
                # plt.close()

        if render:
            plt.close()

        print(f"Episode {episode + 1}, Reward: {episode_reward}")
        total_rewards.append(episode_reward)

    return total_rewards

if __name__ == "__main__":

    config = {
        "test_episodes": 100,
        "recorder": False,
        "recorder_res_path": "comparisons/res/RL_only_res",
        "init_items": [],
        "init_num": [],
        "render": False,
        "stack_size": 1,
        "model_path": "RL_only_stone_pickaxe"
    }


    env = gym.make("MyCrafter-v0")
    if config["recorder"]:
        env = crafter.Recorder(
            env, config["recorder_res_path"],
            save_stats = True,
            save_video = False,
            save_episode = False,
        )

    env = env_wrapper.InitWrapper(env, init_items=config["init_items"], init_num=config["init_num"])
    env = env_wrapper.StoneSwordWrapper(env)
    # env = submodel_wrappers1.stone_pickaxeWrapper(env)

    model = PPO.load(config["model_path"])
    stack_size = config["stack_size"]
    test_episodes = config["test_episodes"]
    render = config["render"]

    total_rewards = test(env, model, test_episodes, render=render, stack_size=stack_size)

    average_reward = sum(total_rewards) / test_episodes
    print(f"Average reward over {test_episodes} episodes: {average_reward}")
