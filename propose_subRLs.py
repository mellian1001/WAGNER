from utils import llm_prompt
from utils import llm_utils
import os
import ast
import json
import argparse


OBJ_INDEX = {"water": 1, "grass": 2, "stone": 3, "path": 4, "sand": 5, "tree": 6, "lava": 7, "coal": 8, "iron": 9, "diamond": 10, "table": 11, "furnace": 12}

ACTION_TABLE = ["Noop", "Move Left", "Move Right", "Move Up", "Move Down", "Do", "Sleep", "Place Stone", "Place Table", "Place Furnace", "Place Plant", "Make Wood Pickaxe", "Make Stone Pickaxe", "Make Iron Pickaxe", "Make Wood Sword", "Make Stone Sword", "Make Iron Sword"]


reward_wrapper_template = """
class {}Wrapper(gym.Wrapper):

    def __init__(self, env, target_obj={}, allowed_actions={}):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
        self.find_target = False
        self.target_obj = target_obj
        self.allowed_actions = allowed_actions
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        self.find_target = False
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        
        if action not in self.allowed_actions:
            reward -= 10

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.001

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)

        for i in range(left_index, right_index, 1):
            if not self.find_target:
                for j in range(up_index, down_index, 1):
                    if (info['semantic'][i][j] == self.target_obj):
                        reward += 5
                        self.find_target = True
                        break

        self.prev_pos = player_pos

        num_item = info["inventory"]["{}"]
        if num_item > self.prev_count:
            reward += 100
            done = True
        self.prev_count = num_item

        return obs, reward, done, info

"""


def define_training_wrapper(obj_name, allowed_actions):

    if obj_name in OBJ_INDEX:
        return reward_wrapper_template.format(obj_name, OBJ_INDEX[obj_name], allowed_actions, obj_name, obj_name)
    else:
        return reward_wrapper_template.format(obj_name, -1, allowed_actions, obj_name, obj_name)


def check_valid(model_info_dict_str):

    valid_items = {"health", "food", "drink", "energy", "sapling", "wood", "stone", "coal", "iron", "diamond", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"}

    try:
        model_info_dict = json.loads(model_info_dict_str)
        if len(model_info_dict) == 0:
            print("no sub model returned! retrying...")
            return False
        if "model0" in model_info_dict.keys():
            print("model name should start with model1, not model0! Retrying....")
            return False
        for value in model_info_dict.values():
            if "name" not in value.keys() or "description" not in value.keys() or "requirement" not in value.keys():
                print("information of sub model is incomplete! Retrying...")
                return False
            if value["name"] not in valid_items:
                print("invalid wrapper name! Retrying...")
                return False
            reqs = value["requirement"]
            for req in reqs:
                if req[0] not in valid_items or len(req) != 2:
                    return False

    except Exception:
        return False
    
    return True


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--rules_path", type=str, default=os.path.join("temp_result", "rules.txt"))
    parser.add_argument("--plan_path", type=str, default=os.path.join("temp_result", "plan.txt"))
    parser.add_argument("--save_model_info_path", type=str)
    parser.add_argument("--final_task", type=str)
    parser.add_argument("--save_wrappers_path", type=str)

    args = parser.parse_args()

    config = {
        "rules_path": args.rules_path,
        "plan_path": args.plan_path,
        "save_model_info": True, 
        "save_model_info_path": args.save_model_info_path,
        "goal": args.final_task,
        "save_wrappers": True,
        "save_wrappers_path": args.save_wrappers_path,
            }

    rules = open(config["rules_path"], 'r').read()
    plan = open(config["plan_path"], 'r').read()
    
    max_retries = 3 
    for i in range(max_retries):

        model_info_dict = llm_utils.llm_chat(prompt = llm_prompt.compose_submodel_prompt(rules, plan, config["goal"]), system_prompt="", model="deepseek-chat")

        is_valid = check_valid(model_info_dict)
        if not is_valid and i == max_retries-1:
            print("LLM output is invalid!")
            assert False
        elif is_valid:
            break

    if config["save_model_info"]:
        with open(config["save_model_info_path"], 'w') as f:
            f.write(model_info_dict)

    print("model information successfully saved at ", config["save_model_info_path"])
    print("models to train: ")
    print(model_info_dict)

    lines = '''import gym
import numpy as np
from gym import spaces


def face_at(obs):

    try:
        return obs.split()[obs.split().index("face") + 1]
    except ValueError as _:
        pass
    return ""

    '''

    if config["save_wrappers"]:

        plan = ast.literal_eval(plan)

        print("saving subtask wrappers...")

        with open(config["save_wrappers_path"], 'w') as f:
            f.write(lines)

        model_info = json.loads(model_info_dict)

        for i, obj_info in enumerate(model_info.values()):

            obj = obj_info["name"] 
            subtask = plan[i]

            with open(config["save_wrappers_path"], 'a') as f:

                for j in range(max_retries):
                    allowed_actions = llm_utils.llm_chat(prompt=llm_prompt.compose_action_prompt(rules, subtask), system_prompt="", model="deepseek-chat")
                    # print(allowed_actions)
                    try:
                        allowed_actions = ast.literal_eval(allowed_actions)
                        allowed_actions = [ACTION_TABLE.index(item) for item in allowed_actions]
                        f.write(define_training_wrapper(obj, allowed_actions))
                        break
                    except Exception:
                        if j == max_retries-1:
                            assert False
                        print("invalid actions format, retrying...")

        print("subtask wrappers successful saved at ", config["save_wrappers_path"])
