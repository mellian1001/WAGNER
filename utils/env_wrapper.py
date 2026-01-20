from . import llm_utils
import gym
import numpy as np
from . import llm_prompt
from gym import spaces

def face_at(obs):

    try:
        return obs.split()[obs.split().index("face") + 1]
    except ValueError as _:
        pass
    return ""


class ActionWrapper(gym.ActionWrapper):

    def __init__(self, env, allowed_actions):
        super(ActionWrapper, self).__init__(env)
        self.allowed_actions = allowed_actions
        self.action_space = spaces.Discrete(len(allowed_actions))

    def action(self, action):
        return self.allowed_actions[action]

    def reverse_action(self, action):
        return self.allowed_actions.index(action)


class LabelGeneratingWrapper(gym.Wrapper):
    """
    This wrapper calls an external function to get a label for each step
    and packages the observation and label into a Dict observation space.
    """
    def __init__(self, env: gym.Env, get_label_func, target_obj, num_aux_classes: int):
        super().__init__(env)
        self.get_label_func = get_label_func
        self.target_obj = target_obj

        # Define the new observation space as a dictionary
        self.observation_space = spaces.Dict({
            'obs': env.observation_space,
            'label': spaces.Discrete(num_aux_classes)
        })

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        label = 0
        dict_obs = {'obs': obs, 'label': label}
        return dict_obs

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        label = self.get_label_func(info['obs'], self.target_obj)
        dict_obs = {'obs': obs, 'label': label}
        return dict_obs, reward, terminated, info


class myFrameStack(gym.Wrapper):

    def __init__(self, env, stack_size=2):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames = []
        
        original_shape = env.observation_space.shape  # (64, 64, 3)
        print(original_shape)
        new_shape = (original_shape[0], original_shape[1], original_shape[2]*stack_size)  # (64, 64, 3*stack_size)
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=new_shape,
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.frames = [obs] * self.stack_size 
        return np.concatenate(self.frames, axis=-1)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.pop(0)
        self.frames.append(obs)
        stacked_obs = np.concatenate(self.frames, axis=-1)
        return stacked_obs, reward, done, info


class RulesWrapper(gym.Wrapper):

    def __init__(self, env, system_prompt, model="deepseek-chat", save=False, save_path="./rules.txt"):
        super().__init__(env)
        self.rule_set = "{}"
        self.cur_step = 0
        self.model = model
        self.save = save
        self.save_path = save_path
        self.system_prompt = system_prompt

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        if reward >= 1 or reward <= -1:
            user_prompt = llm_utils.compose_user_prompt(info["obs"] + info['history'], self.rule_set)
            self.rule_set = llm_utils.llm_chat(user_prompt, system_prompt=self.system_prompt ,model=self.model)
            print("current rules:")
            print(self.rule_set)
            if self.save:
                with open(self.save_path, 'w') as f:
                    f.write(self.rule_set)
        
        self.cur_step += 1

        return obs, reward, done, info

class LLMSubtaskWrapper(gym.Wrapper):

    def __init__(self, env, current_goal, model="qwen2.5:7b"):
        super().__init__(env)
        self.rule_set = "{}"
        self.cur_step = 0
        self.model = model
        self.current_goal = current_goal

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        if reward >= 1 or reward <= -1:
            user_prompt = llm_utils.compose_user_prompt(info["obs"] + info["history"], self.rule_set)
            user_prompt += "\n\nHere is the current_goal: " + self.current_goal
            self.rule_set = llm_utils.llm_chat(user_prompt, system_prompt=llm_prompt.EXPLORATION_PROMPT, model=self.model)
            # if "deepseek" in self.model:
            #     index = self.rule_set.find("</think>")
            #     self.rule_set = self.rule_set[index+8:]
            print(self.rule_set)
        
        self.cur_step += 1

        return obs, reward, done, info

class NavigationWrapper(gym.Wrapper):

    def __init__(self, env, obj_index):
        super().__init__(env)
        self.target_obj = obj_index

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        # reward = 0
        player_pos = info['player_pos']

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)
        
        for i in range(left_index, right_index, 1):
            for j in range(up_index, down_index, 1):
                if (info['semantic'][i][j] == self.target_obj):
                    reward = 1000
                    done = True
                    return obs, reward, done, info
        
        return obs, reward, done, info



class MovementWrapper(gym.Wrapper):

    def __init__(self, env, generate_rule=True):
        super().__init__(env)
        self.prev_pos = np.array([32, 32])

    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        # move = ["move_up", "move_down", "move_left", "move_right"]
        # if info["action"] in move:
        #     reward += 0.1

        play_pos = info["player_pos"]
        if not np.array_equal(play_pos, self.prev_pos):
            reward += 0.1
        else:
            reward -= 0.1

        self.prev_pos = play_pos

        return obs, reward, done, info

class DrinkWaterWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_drink = 10

    def reset(self, **kwargs):
        self.prev_drink = 10
        return self.env.reset()

    def step(self, action):
        
        obs, reward, done, info = self.env.step(action)

        reward = 0
        if info["inventory"]["health"] == 0:
            reward -= 1000

        if info["inventory"]["drink"] > self.prev_drink:
            reward += 1000
            done = True
        self.prev_drink = info["inventory"]["drink"]

        return obs, reward, done, info

class stoneWrapperTest(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_stone = 0
    
    def reset(self, **kwargs):
        self.prev_stone = 0
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        reward = 0

        num_stone = info["inventory"]["stone"]
        if num_stone > self.prev_stone:
            reward += 1
            done = True
        self.prev_stone = num_stone

        return obs, reward, done, info


class MineStoneWrapper2(gym.Wrapper):

    def __init__(self, env, decay_steps=500000, obj_index=3):
        super().__init__(env)
        self.prev_stone = 0
        self.prev_pos = np.array([32, 32])
        self.current_step = 0
        self.decay_steps = decay_steps
        self.find_stone = False
        self.target_obj = obj_index
    
    def reset(self, **kwargs):
        self.prev_stone = 0
        self.find_stone = False
        self.prev_pos = np.array([32, 32])
        return self.env.reset()

    def step(self, action):

        self.current_step += 1

        obs, reward, done, info = self.env.step(action)

        reward = 0
        # 如果agent不动则给予小惩罚
        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.03

        # 如果靠近目标物体则给予适当reward
        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)

        if not self.find_stone:

            for i in range(left_index, right_index, 1):
                for j in range(up_index, down_index, 1):
                    if (info['semantic'][i][j] == self.target_obj):
                        reward += 100
                        self.find_stone = True
                        break

        self.prev_pos = player_pos

        # 如果获得物体则给予大奖励
        num_stone = info["inventory"]["stone"]
        if num_stone > self.prev_stone:
            reward = 10000
            done = True
        self.prev_stone = num_stone

        return obs, reward, done, info


class woodWrapperTest(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_wood = 0
    
    def reset(self, **kwargs):
        self.prev_wood = 0
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        reward = 0

        num_wood = info["inventory"]["wood"]
        if num_wood > self.prev_wood:
            reward = 1
            done = True

        self.prev_wood = num_wood

        return obs, reward, done, info

class wood_pickaxeWrapperTest(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_wood_pickaxe = 0
        self.prev_pos = np.array([32, 32])
        
    
    def reset(self, **kwargs):
        self.prev_wood_pickaxe = 0
        self.prev_pos = np.array([32, 32])
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        reward = 0

        num_wood_pickaxe = info["inventory"]["wood_pickaxe"]
        if num_wood_pickaxe > self.prev_wood_pickaxe:
            reward += 1
            done = True

        self.prev_wood_pickaxe = num_wood_pickaxe

        return obs, reward, done, info

class WoodPickaxeWrapper2(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_wood_pickaxe = 0
        
    
    def reset(self, **kwargs):
        self.prev_wood_pickaxe = 0
        self.prev_pos = np.array([32, 32])
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        reward = 0

        play_pos = info["player_pos"]
        if np.array_equal(play_pos, self.prev_pos):
            reward -= 0.03

        self.prev_pos = play_pos

        num_wood_pickaxe = info["inventory"]["wood_pickaxe"]
        if num_wood_pickaxe > self.prev_wood_pickaxe:
            reward += 1000
            # done = True

        self.prev_wood_pickaxe = num_wood_pickaxe

        return obs, reward, done, info


class stone_pickaxeWrapperTest(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_stone_pickaxe = 0
    
    def reset(self, **kwargs):
        self.prev_stone_pickaxe = 0
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        reward = 0

        num_stone_pickaxe = info["inventory"]["stone_pickaxe"]
        if num_stone_pickaxe > self.prev_stone_pickaxe:
            reward += 1
            done = True

        self.prev_stone_pickaxe = num_stone_pickaxe

        return obs, reward, done, info

class InitWrapper(gym.Wrapper):

    def __init__(self, env, init_items=[], init_num=[], init_center="mid"):
        super().__init__(env)
        self.init_items = init_items
        self.init_num = init_num
        self.init_center = init_center

    def reset(self, **kwargs):

        self.env.reset_aux(init_items=self.init_items, init_num=self.init_num, init_center=self.init_center)
        return self.env.reset()

class MineCoalWrapper(gym.Wrapper):

    def __init__(self, env, navigation_model):
        super().__init__(env)
        self.model = navigation_model
        self.prev_coal = 0

    def reset(self, **kwargs):

        self.prev_coal = 0
        
        obs = self.env.reset(**kwargs)

        valid = False

        for _ in range(100):

            if not valid:

                action, _  = self.model.predict(obs, deterministic=True)
                obs, _, _, info = self.env.step(action)

                player_pos = info['player_pos']

                left_index = max(0, player_pos[0] - 4)
                right_index = min(64, player_pos[0] + 4)
                up_index = max(0, player_pos[1] - 3)
                down_index = min(64, player_pos[1] + 3)
                
                for i in range(left_index, right_index, 1):
                    if not valid:
                        for j in range(up_index, down_index, 1):
                            if (info['semantic'][i][j] == 8):
                                valid = True
                                break
        return obs
    
    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        # reward = 0
        num_coal = info["inventory"]["coal"]

        if num_coal > self.prev_coal:
            reward += 1000
            done = True
        self.prev_coal = num_coal

        return obs, reward, done, info

class FurnaceWrapper(gym.Wrapper):

    def __init__(self, env):
        self.prev_pos = np.array([32, 32])
        super().__init__(env)
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
 
        play_pos = info["player_pos"]
        if np.array_equal(play_pos, self.prev_pos):
            reward -= 0.03

        self.prev_pos = play_pos

        if face_at(info["obs"]) == "furnace":
            reward += 1000
            done = True

        return obs, reward, done, info

class MineIronWrapper(gym.Wrapper):

    def __init__(self, env, navigation_model):
        super().__init__(env)
        self.model = navigation_model
        self.prev_iron = 0

    def reset(self, **kwargs):

        self.prev_iron = 0
        
        obs = self.env.reset(**kwargs)

        valid = False

        for _ in range(100):

            if not valid:

                action, _  = self.model.predict(obs, deterministic=True)
                obs, _, _, info = self.env.step(action)

                player_pos = info['player_pos']

                left_index = max(0, player_pos[0] - 4)
                right_index = min(64, player_pos[0] + 4)
                up_index = max(0, player_pos[1] - 3)
                down_index = min(64, player_pos[1] + 3)
                
                for i in range(left_index, right_index, 1):
                    if not valid:
                        for j in range(up_index, down_index, 1):
                            if (info['semantic'][i][j] == 9):
                                valid = True
                                break
        return obs
    
    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        # reward = 0
        num_iron = info["inventory"]["iron"]

        if num_iron > self.prev_iron:
            reward += 1000
            done = True
        self.prev_iron = num_iron

        return obs, reward, done, info


class MineCoalWrapper2(gym.Wrapper):

    def __init__(self, env, obj_index=8):
        super().__init__(env)
        self.target_obj = obj_index
        self.prev_coal = 0
        self.find_coal = False
        self.prev_pos = np.array([32, 32])

    def reset(self, **kwargs):
        self.prev_coal = 0
        self.find_coal = False
        self.prev_pos = np.array([32, 32])
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        # reward = 0
        player_pos = info['player_pos']

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)
        
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.03

        self.prev_pos = player_pos

        if not self.find_coal:

            for i in range(left_index, right_index, 1):
                for j in range(up_index, down_index, 1):
                    if (info['semantic'][i][j] == self.target_obj):
                        reward += 100
                        self.find_coal = True
                        break
        
        num_coal = info["inventory"]["coal"]
        if num_coal > self.prev_coal:
            reward += 10000
            done = True
        self.prev_coal = num_coal
        
        return obs, reward, done, info

class MineIronWrapper2(gym.Wrapper):

    def __init__(self, env, obj_index=9):
        super().__init__(env)
        self.target_obj = obj_index
        self.prev_iron = 0
        self.find_iron = False
        self.prev_pos = np.array([32, 32])

    def reset(self, **kwargs):
        self.prev_iron = 0
        self.find_iron = False
        self.prev_pos = np.array([32, 32])
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        # reward = 0
        player_pos = info['player_pos']

        left_index = max(0, player_pos[0] - 4)
        right_index = min(64, player_pos[0] + 4)
        up_index = max(0, player_pos[1] - 3)
        down_index = min(64, player_pos[1] + 3)
        
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.03

        self.prev_pos = player_pos

        if not self.find_iron:

            for i in range(left_index, right_index, 1):
                for j in range(up_index, down_index, 1):
                    if (info['semantic'][i][j] == self.target_obj):
                        reward += 100
                        self.find_iron = True
                        break
        
        num_iron = info["inventory"]["iron"]
        if num_iron > self.prev_iron:
            reward += 10000
            done = True
        self.prev_iron = num_iron
        
        return obs, reward, done, info

class WrongWrapperTest(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_stone_pickaxe = 0
        self.prev_stone = 0
        self.prev_wood = 0
        self.prev_wood_pickaxe = 0
        self.prev_iron = 0
        self.prev_coal = 0
        self.prev_iron_pickaxe = 0
    
    def reset(self, **kwargs):
        self.prev_stone_pickaxe = 0
        self.prev_stone = 0
        self.prev_wood = 0
        self.prev_wood_pickaxe = 0
        self.prev_iron = 0
        self.prev_coal = 0
        self.prev_iron_pickaxe = 0
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        reward = 0

        num_stone_pickaxe = info["inventory"]["stone_pickaxe"]
        num_stone = info["inventory"]["stone"]
        num_wood = info["inventory"]["wood"]
        num_wood_pickaxe = info["inventory"]["wood_pickaxe"]
        num_iron = info["inventory"]["iron"]
        num_coal = info["inventory"]["coal"]
        num_iron_pickaxe = info["inventory"]["iron_pickaxe"]
    
        if num_stone_pickaxe > self.prev_stone_pickaxe and num_stone_pickaxe == 1:
            reward += 1
        if num_wood_pickaxe > self.prev_wood_pickaxe and num_wood_pickaxe == 1:
            reward += 1
        if num_stone > self.prev_stone and num_stone == 1:
            reward += 1
        if num_wood > self.prev_wood and num_wood == 1:
            reward += 1
        if num_iron > self.prev_iron and num_iron == 1:
            reward += 1
        if num_coal > self.prev_coal and num_coal == 1:
            reward += 1
        if num_iron_pickaxe > self.prev_iron_pickaxe and num_iron_pickaxe == 1:
            reward += 1

        self.prev_stone_pickaxe = num_stone_pickaxe
        self.prev_wood_pickaxe = num_wood_pickaxe
        self.prev_wood = num_wood
        self.prev_stone = num_stone
        self.prev_iron = num_iron
        self.prev_coal = num_coal
        self.prev_iron_pickaxe = num_iron_pickaxe

        return obs, reward, done, info

class WrapperTest(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.achieved = set()  # 记录已经获得过的物品
    
    def reset(self, **kwargs):
        self.achieved = set()
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = 0

        items_to_check = {
            'stone_pickaxe': info["inventory"]["stone_pickaxe"],
            'wood_pickaxe': info["inventory"]["wood_pickaxe"],
            'stone': info["inventory"]["stone"],
            'wood': info["inventory"]["wood"],
            'iron': info["inventory"]["iron"],
            'coal': info["inventory"]["coal"],
            'iron_pickaxe': info["inventory"]["iron_pickaxe"]
        }
        
        for item_name, count in items_to_check.items():
            if count > 0 and item_name not in self.achieved:
                reward += 1
                self.achieved.add(item_name)

        return obs, reward, done, info

class coalWrapperTest(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_coal = 0
    
    def reset(self, **kwargs):
        self.prev_coal = 0
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        reward = 0

        num_coal = info["inventory"]["coal"]
        if num_coal > self.prev_coal:
            reward += 1
            done = True
        self.prev_coal = num_coal

        return obs, reward, done, info
    
class iron_pickaxeWrapperTest(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_iron_pickaxe = 0
    
    def reset(self, **kwargs):
        self.prev_iron_pickaxe = 0
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        reward = 0

        num_iron_pickaxe = info["inventory"]["iron_pickaxe"]
        if num_iron_pickaxe > self.prev_iron_pickaxe:
            reward += 1
            done = True
        self.prev_iron_pickaxe = num_iron_pickaxe

        return obs, reward, done, info