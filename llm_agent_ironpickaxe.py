import matplotlib.pyplot as plt
import time
from PIL import Image
import gym
from stable_baselines3 import PPO
from crafter import crafter
from utils import env_wrapper
import os
from utils import llm_prompt
from utils import llm_utils
import numpy as np
from tqdm import tqdm
from model import CustomACPolicy
import json
import ast


ITEM_TABLE = ["sapling", "wood", "stone", "coal", "iron", "diamond", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"]
OBJ_TABLE = ["furnace", "table"]

def show(fig, image_display, obs):

    img = Image.fromarray(obs)

    image_display.set_data(img)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    time.sleep(0.2)

def get_item_number(info, item):

    return info["inventory"][item]

def create_model_description(available_models, submodels):

    description = {}
    for model in available_models:
        if model == "model0":
            continue
        description[model] = submodels[model]["description"]

    return str(description)

def available_models(info, submodels):

    available_models = ["model0"]
    
    for model_name, submodel in submodels.items():
        if submodel["requirement"] == []:
            available_models.append(model_name)
        else:
            flag = True
            for requirement in submodel["requirement"]:
                if get_item_number(info, requirement[0]) < requirement[1]:
                    flag = False     
            if flag:
                available_models.append(model_name)
    return available_models


def choose_model(goal, info, last_model_call, model_list, rules, model_description):
    llm_response = llm_utils.llm_chat(model="qwen2.5-32b-instruct", prompt=llm_prompt.compose_llm_agent_prompt(rules=rules, model_description=model_description, current_goal=goal, info=info, last_model_call=last_model_call), system_prompt="")
    #llm_response = llm_utils.llm_chat(model="deepseek-chat", prompt=llm_prompt.compose_llm_agent_prompt(rules=rules, model_description=model_description, current_goal=goal, info=info, last_model_call=last_model_call), system_prompt="")

    print(llm_response)

    for key in model_list.keys():
        if "call " + str(key) in llm_response:
            model = model_list[key]
            print("calling " + key)
            return model, key

    return base_model, "model0"

def face_at(obs):

    try:
        return obs.split()[obs.split().index("face") + 1]
    except ValueError as _:
        pass
    return ""

def is_finished(info, last_info, submodels):

    for item in submodels.values():
        if item["name"] in ITEM_TABLE:
            if info["inventory"][item["name"]] > last_info["inventory"][item["name"]]:
                return True
        elif item["name"] in OBJ_TABLE:
            if face_at(info["obs"]) == item["name"] and info["achievements"]["place_"+item["name"]] == 0:
                return True
        else:
            assert False
    
    return False

def is_current_goal_achieved(goal, info, last_info):

    if goal in ITEM_TABLE:
        return get_item_number(info, goal) > get_item_number(last_info, goal)
    elif goal in OBJ_TABLE:
        return face_at(info["obs"]) == goal
    
    print("invalid goal!")
    assert False


def not_moved(prev_locations):
    
    for i in range(1, len(prev_locations)):
        if not np.array_equal(prev_locations[i], prev_locations[i-1]):
            return False
    return True


def test(env, model_list, num_episodes, rules, model_description, goal_list, plan_list, submodels, stack_size=1, last_model_call="", render=True):

    if goal_list == []:
        print("Please make sure there is at least one goal in goal_list")
        return 

    num_goals = len(goal_list)

    print("Testing...")

    total_rewards = []

    for episode in tqdm(range(num_episodes)):

        index = 0

        obs = env.reset()

        if render:

            plt.ion()
            fig, ax = plt.subplots()
            image_display = ax.imshow(obs)
            plt.show(block=False)

        done = False
        episode_reward = 0

        frames = [obs] * stack_size 
        model = model_list["model0"]
        model_name = "model0"
        last_model_call = ""
        last_info = ""
        prev_locations = [np.array([0, 0])] * 5 + [np.array([1, 1])] * 5

        is_first_epoch = True

        while not done:

            if not_moved(prev_locations):
                action = np.random.choice([1, 2, 3, 4])
            else:
                action, _ = model.predict(np.concatenate(frames, axis=-1), deterministic=False)
            frames.pop(0)
            frames.append(obs)

            obs, reward, done, info = env.step(action)

            prev_locations.pop(0)
            prev_locations.append(info["player_pos"])

            if is_first_epoch:

                is_first_epoch = False
                last_info = info
                continue

            if index != num_goals and is_current_goal_achieved(goal_list[index], info, last_info):

                index += 1

            if index >= num_goals:

                model = base_model

            else:

                if is_finished(info, last_info, submodels):
                    print("current_goal: ", goal_list[index])

                    model_description = create_model_description(available_models(info, submodels), submodels)
                    # print(model_description)

                    model, model_name = choose_model(plan_list[index], info, last_model_call, model_list, rules, model_description)
                    last_model_call = model_name

            last_info = info
            episode_reward += reward

            if render:
                
                show(fig, image_display, obs)
        print(f"Episode {episode + 1}, Reward: {episode_reward}")
        total_rewards.append(episode_reward)

    return total_rewards

if __name__ == "__main__":

    config = {
        "test_episodes": 200, #SET_TEST_EPISODES_NUM
        "recorder": True, 
        "recoder_res_path": "iron_pickaxe/res/qwen", #SET_SAVE_PATH
        "init_items": [],
        "init_num": [],
        "render": False,
        "goal_list_path": os.path.join("iron_pickaxe/temp_result", "goal_list.txt"),
        "plan_path": os.path.join("iron_pickaxe/temp_result", "plan1.txt"),
        "stack_size": 1,
        "submodels_path": "RL_models2",
        "model_info_dict_path": os.path.join("iron_pickaxe/temp_result", "model_info.json"),
        "rules_path": os.path.join("iron_pickaxe/temp_result", "human_designed_rules.txt"),
    }

    env = gym.make("MyCrafter-v0")

    if config["recorder"]:
        env = crafter.Recorder(
            env, config["recoder_res_path"],
            save_stats = True,
            save_video = False,
            save_episode = False,
        )
    env = env_wrapper.InitWrapper(env, init_items=config["init_items"], init_num=["init_num"])

    with open (config["model_info_dict_path"], 'r') as f:
        submodels = json.load(f)

    model_description = str(submodels)
    rules = open(config["rules_path"], 'r').read()

    model_list = {}

    base_model = PPO.load(os.path.join("RL_models2", "wood"))

    model_list["model0"] = base_model
    for model_name, submodel in submodels.items():

        model = PPO.load(os.path.join(config["submodels_path"], submodel["name"]))
        model_list[model_name] = model

test_episodes = config["test_episodes"]
render = config["render"]
stack_size = config["stack_size"]

goal_list_path = config["goal_list_path"]
with open(goal_list_path) as f:
    goal_list_string = f.read()
goal_list = ast.literal_eval(goal_list_string) 

plan_path = config["plan_path"]
with open(plan_path) as f:
    plan_string = f.read()
plan_list = ast.literal_eval(plan_string)


total_rewards = test(env, model_list, test_episodes, rules=rules, submodels=submodels, model_description=model_description, goal_list=goal_list, plan_list=plan_list, render=render, last_model_call="", stack_size=stack_size)

average_reward = sum(total_rewards) / test_episodes
print(f"Average reward over {test_episodes} episodes: {average_reward}")