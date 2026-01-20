import gym
import numpy as np

def face_at(obs):

    try:
        return obs.split()[obs.split().index("face") + 1]
    except ValueError as _:
        pass
    return ""


class wood_pickaxeWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.03
            # done = True
            self.prev_pos = player_pos
        try:
            num_item = info["inventory"]["wood_pickaxe"]
            if num_item > self.prev_count:
                reward += 1000
                # done = True
                self.prev_count = num_item
        except KeyError as e:
            if face_at(info["obs"]) == "wood_pickaxe":
                reward += 1000

        return obs, reward, done, info


class stoneWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.03

        self.prev_pos = player_pos
        try:
            num_item = info["inventory"]["stone"]
            if num_item > self.prev_count:
                reward += 1000
                # done = True
                self.prev_count = num_item
        except KeyError as e:
            if face_at(info["obs"]) == "stone":
                reward += 1000

        return obs, reward, done, info


class stone_pickaxeWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.03

        self.prev_pos = player_pos
        try:
            num_item = info["inventory"]["stone_pickaxe"]
            if num_item > self.prev_count:
                reward += 1000
                # done = True
                self.prev_count = num_item
        except KeyError as e:
            if face_at(info["obs"]) == "stone_pickaxe":
                reward += 1000

        return obs, reward, done, info


class coalWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.03

        self.prev_pos = player_pos
        try:
            num_item = info["inventory"]["coal"]
            if num_item > self.prev_count:
                reward += 1000
                # done = True
                self.prev_count = num_item
        except KeyError as e:
            if face_at(info["obs"]) == "coal":
                reward += 1000

        return obs, reward, done, info


class furnaceWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.03

        self.prev_pos = player_pos
        try:
            num_item = info["inventory"]["furnace"]
            if num_item > self.prev_count:
                reward += 1000
                # done = True
                self.prev_count = num_item
        except KeyError as e:
            if face_at(info["obs"]) == "furnace":
                reward += 1000

        return obs, reward, done, info


class ironWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_count = 0
        self.prev_pos = np.array([32, 32])
    
    def reset(self, **kwargs):
        self.prev_pos = np.array([32, 32])
        self.prev_count = 0
        return self.env.reset()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        player_pos = info["player_pos"]
        if np.array_equal(player_pos, self.prev_pos):
            reward -= 0.03

        self.prev_pos = player_pos
        try:
            num_item = info["inventory"]["iron"]
            if num_item > self.prev_count:
                reward += 1000
                # done = True
                self.prev_count = num_item
        except KeyError as e:
            if face_at(info["obs"]) == "iron":
                reward += 1000

        return obs, reward, done, info

