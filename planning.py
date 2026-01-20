from utils import llm_prompt
from utils import llm_utils
import ast
import os
import argparse


VALID_ITEMS = {"health", "food", "drink", "energy", "sapling", "wood", "stone", "coal", "iron", "diamond", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"}


def plan(tasks_list, num_step, rules):

    print("planning...")

    return plan_aux(tasks_list, [], 0, num_step, rules)


def plan_aux(tasks_list, goal_list, current_step, num_step, rules):

    if current_step == num_step or len(tasks_list) == 0:
        return tasks_list, goal_list

    current_tasks_list = tasks_list

    tasks_list = []
    goal_list = []

    for subgoal in current_tasks_list:

        PLANNING_PROMPT = llm_prompt.compose_planning_prompt(rules)

        response = llm_utils.llm_chat(prompt="Current goal: " + subgoal, system_prompt=PLANNING_PROMPT, model="deepseek-chat")
        try: 
            llm_subgoals_list = ast.literal_eval(response)
            for new_subgoal in llm_subgoals_list:
                response = llm_utils.llm_chat(prompt=new_subgoal,system_prompt=llm_prompt.TRANS_PROMPT, model="deepseek-chat")
                if response in VALID_ITEMS and response not in goal_list:
                    tasks_list.append(new_subgoal)
                    goal_list.append(response)

        except Exception:
            pass

    print(tasks_list)
    print(goal_list)

    return plan_aux(tasks_list, goal_list, current_step+1, num_step, rules)


def check_valid(goal_list, tasks_list):
    
    VALID_ITEMS = {"health", "food", "drink", "energy", "sapling", "wood", "stone", "coal", "iron", "diamond", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"}

    try:
        if len(goal_list) != len(tasks_list):
            return False
        for goal in goal_list:
            if goal not in VALID_ITEMS:
                return False
    except Exception:
        return False

    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--rules_path", type=str, default=os.path.join("temp_result", "human_designed_rules.txt"))
    parser.add_argument("--final_task", type=str, default="collect an iron")
    parser.add_argument("--save_plan_path", type=str)
    parser.add_argument("--save_goal_list_path", type=str)

    args = parser.parse_args()

    config = {"rules_path": args.rules_path,
              "final_task":  args.final_task,
              "planning_steps": 2,
              "save_plan": True,
              "save_goal_list": True,
              "save_plan_path": args.save_plan_path,
              "save_goal_list_path": args.save_goal_list_path,
              }

    rules = open(config["rules_path"], 'r').read()
    final_task = config["final_task"]
    tasks_list = [final_task]
    planning_steps = config["planning_steps"]
    
    max_retries = 3
    for i in range(max_retries):
        tasks_list, goal_list = plan(tasks_list, planning_steps, rules)
        is_valid = check_valid(goal_list, tasks_list)
        if not is_valid and i == max_retries-1:
            print("LLM output is invalid!")
            assert False
        elif is_valid:
            break
        print("found LLM ouput invalid, retrying...")

    # print(tasks_list)
    # print(goal_list)

    if config["save_plan"]:

        with open(config["save_plan_path"], 'w') as f:
            f.write(str(tasks_list))
        print("plan successful saved at ", config["save_plan_path"])

    if config["save_goal_list"]:

        with open(config["save_goal_list_path"], 'w') as f:
            f.write(str(goal_list))
        print("goal list successful saved at ", config["save_goal_list_path"])

        print("LLM generated goals: ")
        print(goal_list)
