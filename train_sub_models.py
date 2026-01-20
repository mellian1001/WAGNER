import gym
from model import CustomResNet, CustomACPolicy, CustomPPO
import torch.nn as nn
import torch
import json
import crafter
from utils import env_wrapper
import os
from stable_baselines3 import PPO
import importlib
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import warnings

warnings.filterwarnings("ignore")


def train_model(env, save_path, config, total_timesteps=1000000):

    policy_kwargs = {
        "features_extractor_class": CustomResNet,
        "features_extractor_kwargs": {"features_dim": 1024},
        "activation_fn": nn.ReLU,
        "net_arch": [],
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {"eps": 1e-5}
    }

    model = CustomPPO(
        CustomACPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=512,
        n_epochs=3,
        gamma=0.95,
        gae_lambda=0.65,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        normalize_advantage=False
    )

    total_timesteps = total_timesteps

    call_back_on_best = StopTrainingOnRewardThreshold(reward_threshold=config["stop_threshold"], verbose=1)

    eval_callback = EvalCallback(
        env,
        callback_on_new_best=call_back_on_best,
        eval_freq=20000,
        verbose=1,
        n_eval_episodes=30,
        deterministic=False,
    )

    if config["early_stop"]:
        model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback)
    else:
        model.learn(total_timesteps=total_timesteps, progress_bar=True)

    if config["save_model"]:
        model.save(save_path)
        print(f"model successfully saved at {save_path}!")

    env.close()


def import_subtask_wrappers(wrapper_path="submodel_wrappers.py"):

    wrappers = importlib.import_module(wrapper_path)
    return wrappers


if __name__ == "__main__":


    config = {
        "wrapper_path": "temp_result.submodel_wrappers1", 
        "model_info_path": os.path.join("temp_result", "submodels1.json"),
        "save_model": True,
        "model_save_path": "RL_models2",
        "total_timesteps": 1000000,
        "early_stop": True,
        "stop_threshold": 90,
        "action_shaping": False,
    }

    with open(config["model_info_path"]) as f:
        model_info = json.load(f)

    wrappers = import_subtask_wrappers(config["wrapper_path"])

    for i, model_dict in enumerate(model_info.values()):

        name = model_dict["name"]

        print(f"training sub model: {name} ({i + 1} / {len(model_info)})...\n")

        reward_wrapper_name = name + "Wrapper"
        requirements = model_dict["requirement"]

        env = gym.make("MyCrafter-v0")

        try:
            wrapper = getattr(wrappers, reward_wrapper_name)
            if config["action_shaping"] is False:
                env = wrapper(env, allowed_actions=range(17))
            else:
                env = wrapper(env)
        except Exception as e:
            print("ERROR! {} is not defined".format(reward_wrapper_name))
            assert False

        init_items = []
        init_num = []
        if requirements != []:
            for r in requirements:
                init_items.append(r[0])
                init_num.append(r[1])

        env = env_wrapper.InitWrapper(env, init_items, init_num)

        save_path = os.path.join(config["model_save_path"], model_dict["name"])

        if not os.path.exists(config["model_save_path"]):
            os.mkdir(config["model_save_path"])

        print("initial items: {}".format(init_items))
        print("initial counts: {}".format(init_num))

        train_model(env, config=config, save_path=save_path, total_timesteps=config["total_timesteps"])
