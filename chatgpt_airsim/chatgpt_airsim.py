import openai
import re
import argparse
from airsim_wrapper import *
import math
import numpy as np
import os
import json
import time

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="prompts/airsim_detection.txt")
parser.add_argument("--sysprompt", type=str, default="system_prompts/airsim_basic.txt")
parser.add_argument("--vllm-url", type=str, default="http://192.9.203.198:8000/v1", 
                    help="VLLM server URL")
parser.add_argument("--model", type=str, default="mistralai/Ministral-3-14B-Instruct-2512",
                    help="Model name served by VLLM")
parser.add_argument("--ground-level", type=float, default=20.0,
                    help="Ground level offset in meters (distance from Z=0 to ground)")
args = parser.parse_args()

with open("config.json", "r") as f:
    config = json.load(f)

print("Initializing VLLM client...")
# VLLM 서버에 연결
openai.api_key = "EMPTY"
openai.api_base = args.vllm_url

with open(args.sysprompt, "r") as f:
    sysprompt = f.read()

chat_history = [
    {
        "role": "system",
        "content": sysprompt
    },
    {
        "role": "user",
        "content": "move 10 units up"
    },
    {
        "role": "assistant",
        "content": """```python
aw.fly_to([aw.get_drone_position()[0], aw.get_drone_position()[1], aw.get_drone_position()[2]+10])
```
This code uses the `fly_to()` function to move the drone to a new position that is 10 units up from the current position. It does this by getting the current position of the drone using `get_drone_position()` and then creating a new list with the same X and Y coordinates, but with the Z coordinate increased by 10. The drone will then fly to this new position using `fly_to()`."""
    }
]

def ask(prompt):
    chat_history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    completion = openai.ChatCompletion.create(
        model=args.model,
        messages=chat_history,
        temperature=0.1,
        max_tokens=2048
    )
    chat_history.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )
    return chat_history[-1]["content"]

print(f"Done.")

code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)

def extract_python_code(content):
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)
        if full_code.startswith("python"):
            full_code = full_code[7:]
        return full_code
    else:
        return None

class colors:
    RED = "\033[31m"
    ENDC = "\033[m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"

print(f"{colors.GREEN}Initializing AirSim...{colors.ENDC}")
aw = AirSimWrapper(ground_level_offset=args.ground_level)
print(f"{colors.GREEN}Done.{colors.ENDC}")

with open(args.prompt, "r") as f:
    prompt = f.read()

ask(prompt)
print("Welcome to the AirSim chatbot! I am ready to help you with your AirSim questions and commands.")

while True:
    question = input(colors.YELLOW + "AirSim> " + colors.ENDC)
    
    if question == "!quit" or question == "!exit":
        break
    
    if question == "!clear":
        os.system("cls" if os.name == "nt" else "clear")
        continue
    
    response = ask(question)
    print(f"\n{response}\n")
    
    code = extract_python_code(response)
    if code is not None:
        print("Please wait while I run the code in AirSim...")
        try:
            exec(code)
            print("Done!\n")
        except Exception as e:
            print(f"{colors.RED}Error executing code: {e}{colors.ENDC}\n")