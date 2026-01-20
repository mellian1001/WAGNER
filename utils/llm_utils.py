from ollama import chat
from openai import OpenAI
import os


def llm_chat(prompt, system_prompt = "", model="deepseek-chat"):

    messages = []

    sys_prompt = {}
    sys_prompt['role'] = "system"
    sys_prompt['content'] = system_prompt

    messages.append(sys_prompt)

    user_prompt = {}
    user_prompt['role'] = "user"
    user_prompt['content'] = prompt

    messages.append(user_prompt)

    if model == "deepseek-chat":

        api_key = os.getenv("DEEPSEEK_API_KEY")

        if not api_key:
            raise ValueError("No DEEPSEEK_API_KEY is defined")

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        # response = client.chat.completions.create(model = model, messages = messages, stream = False, seed=100)
        response = client.chat.completions.create(model = model, messages = messages, stream = False)
        text = response.choices[0].message.content

    else:

        # assert False

        #response = chat(model=model, messages=messages)
        #text = response["message"]["content"]
        api_key = os.getenv("DASHSCOPE_API_KEY")

        if not api_key:
            raise ValueError("No DASHSCOPE_API_KEY is defined")

        client = OpenAI(api_key=api_key, 
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        # response = client.chat.completions.create(model = model, messages = messages, stream = False, seed=100)
        response = client.chat.completions.create(model = model, 
                                                  messages = messages, 
                                                  #extra_body={"enable_thinking": False},
                                                  stream = False)
        text = response.choices[0].message.content

    return text


def compose_user_prompt(obs, rule_set):

    return "Here is your observation:\n\n" + obs + "\n\n Here is current rule set:\n\n" + rule_set


