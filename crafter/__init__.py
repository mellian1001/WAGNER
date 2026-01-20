from gym.envs.registration import register

from .crafter_env import Crafter

register(
    id = 'MyCrafter-v0',
    entry_point = "crafter.crafter_env:Crafter",
    kwargs={'reward': True}
)

from .crafter_easy_env import Crafter_easy

register(
    id = 'MyCrafter-v1',
    entry_point = "crafter.crafter_easy_env:Crafter_easy",
    kwargs={'reward': True}
)
