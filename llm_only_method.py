import os
os.environ["MINEDOJO_HEADLESS"]="1"
import argparse
import numpy as np
from tqdm import tqdm
import gym
from ollama import chat
from ollama import ChatResponse
from crafter import crafter


parser = argparse.ArgumentParser()
parser.add_argument('--llm_name', type=str, default='qwen2.5:32b', help='Name of the LLM')
parser.add_argument('--env_names', type=str, default=None, help='Comma separated list of environments to run')

args = parser.parse_args()


LLM_name="PUT_LLM_NAME_HERE"

# Replace with your own LLM API.
# Note: query_model takes two arguments: 1) message in openai chat completion form (list of dictionaries), 
#                                        2) an index to indicate where the message should be truncated if the length exceeds LLM context length.

def query_model(messages, index):

    response: ChatResponse = chat(model=LLM_name, messages=messages)

    # print("########################################################################")
    # print(response.message.content)
    # print("########################################################################")

    return response.message.content



def compose_ingame_prompt(info, question, past_qa=[]):
    messages = [
        {"role": "system", "content" : "You’re a player trying to play the game."}
    ]
    
    if len(info['manual'])>0:
        messages.append({"role": "system", "content": info['manual']}) 

    if len(info['history'])>0:
        messages.append({"role": "system", "content": info['history']})

    messages.append({"role": "system", "content": "current step observation: {}".format(info['obs'])})

    if len(past_qa)>0:
        for q,a in past_qa:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": question})

    return messages, 2 # This is the index of the history, we will truncate the history if it is too long for LLM

questions=[
        "What is the best action to take? Let's think step by step, ",
        "Choose the best executable action from the list of all actions. Write the exact chosen action."
    ]

config = {"recorder": True,
          "recorder_res_path": "y_gen_on_other_tasks/comparisons/res/0116test",
          "num_iter": 10}


def run(env_name):
    normalized_scores = []
    env = gym.make("MyCrafter-v0")
    if config["recorder"]:
        env = crafter.Recorder(
            env, config["recorder_res_path"],
            save_stats = True,
            save_video = False,
            save_episode = False,
        )
    env_steps = env.default_steps
    # num_iter = env.default_iter
    num_iter = config["num_iter"]

    def match_act(output):
        inds = [(i, output.lower().index(act.lower())) for i, act in enumerate(env.action_list) if act.lower() in output.lower()]
        if len(inds)>0:
            # return the action with smallest index
            return sorted(inds, key=lambda x:x[1])[0][0]
        else:
            # print("LLM failed with output \"{}\", taking action 0...".format(output))
            return 0

    rewards = []
    progresses = []
    for eps in tqdm(range(num_iter), desc="Evaluating LLM {} on {}".format(LLM_name, env_name)):
        step = 0
        trajectories = []
        qa_history = []
        progress = [0]
        reward = 0
        rewards = []
        done=False

        columns=["Context", "Step", "OBS", "History", "Score", "Reward", "Total Reward"] + questions + ["Action"]

        env.reset()
        info = ""
        
        while step < env_steps:

            if info == "":
                _, reward, done, info = env.step(0)
                step += 1
                continue

            new_row = [info['manual'], step, info['obs'], info['history'], info['score'], reward, sum(rewards)]
            
            if done:
                break
            
            qa_history = []
            for question in questions:
                prompt = compose_ingame_prompt(info, question, qa_history)
                # print("########################################################################")
                # print(prompt)
                # print("########################################################################")
                answer = query_model(*prompt)
                qa_history.append((question, answer))
                new_row.append(answer)
                answer_act = answer

            a = match_act(answer_act)
            print(a)
            new_row.append(env.action_list[a])
            _, reward, done, info = env.step(a)
            rewards.append(reward)
            score=info['score']

            step += 1

        if not done:
            completion=0
        else:
            completion=info['completed']
        progresses.append(np.max(progress))
    return score

env_name = "crafter"

score = run(env_name)

print("Score on Crafter", score)
