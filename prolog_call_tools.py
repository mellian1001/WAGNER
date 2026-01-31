import warnings
warnings.filterwarnings('ignore')


import gym
from stable_baselines3 import PPO
from crafter import crafter
import os


def init_env(env_name):

    global env
    global config
    global obs

    env = gym.make(env_name)
    config = {
        "submodels_path": "RL_models2",
    }
    obs = env.reset()

    return "true"

def call_wood(Env):

    global env
    global config
    global obs

    wood_model = PPO.load(os.path.join(config["submodels_path"], "wood"))

    done = False
    while not done:

        action, _ = wood_model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)

        if info["inventory"]["wood"] == 1:
            return "true"

    return "false"

def call_wood_pickaxe(Env):

    global env
    global config
    global obs

    wood_pickaxe_model = PPO.load(os.path.join(config["submodels_path"], "wood_pickaxe"))

    done = False
    while not done:

        action, _ = wood_pickaxe_model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)

        if info["inventory"]["wood_pickaxe"] == 1:
            return "true"
    
    return "false"

def call_stone(Env):

    global env
    global config
    global obs

    wood_pickaxe_model = PPO.load(os.path.join(config["submodels_path"], "stone"))

    done = False
    while not done:

        action, _ = wood_pickaxe_model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)

        if info["inventory"]["stone"] == 1:
            return "true"
    
    return "false"

def call_stone_pickaxe(Env):

    global env
    global config
    global obs

    wood_pickaxe_model = PPO.load(os.path.join(config["submodels_path"], "stone_pickaxe"))

    done = False
    while not done:

        action, _ = wood_pickaxe_model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)

        if info["inventory"]["stone_pickaxe"] == 1:
            return "true"
    
    return "false"

# init_env("MyCrafter-v0")
# print(call_wood("MyCrafter-v0"))