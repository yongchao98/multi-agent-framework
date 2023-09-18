from prompt_env3 import *
from LLM import *
from sre_constants import error
import random
import os
import json
import re
import copy
import numpy as np
import shutil
import time
import ast

def state_update_func(pg_dict, lifter_weight_list):
  volume_list = [volume for volume, weight in pg_dict.items()]

  state_update_prompt = f'The left boxes in the warehouse are: '
  left_box = ''
  for i in range(len(volume_list)-1):
    state_update_prompt += f'box[{volume_list[i]}V], '
    left_box += f'box[{volume_list[i]}V], '
  state_update_prompt += f'box[{volume_list[len(volume_list)-1]}V]'
  left_box += f'box[{volume_list[len(volume_list)-1]}V]'
  state_update_prompt += f'.\n'
  left_box += f'.\n'

  state_update_prompt += f'The available lifting agents in the warehouse are: '
  for i in range(len(lifter_weight_list)-1):
    state_update_prompt += f'agent[{lifter_weight_list[i]}W], '
  state_update_prompt += f'agent[{lifter_weight_list[len(lifter_weight_list)-1]}W]'
  state_update_prompt += f'.\n'
  return state_update_prompt, left_box

def with_action_syntactic_check_func(pg_dict_input, response, user_prompt_list_input, response_total_list_input, model_name, dialogue_history_method):
  user_prompt_list = copy.deepcopy(user_prompt_list_input)
  response_total_list = copy.deepcopy(response_total_list_input)
  iteration_num = 0
  token_num_count_list_add = []
  while iteration_num < 6:
    response_total_list.append(response)
    feedback = ''
    #try:
    original_response_dict = json.loads(response)
    pg_dict_original = copy.deepcopy(pg_dict_input)

    # The state to be updated
    volume_list = [volume for volume, weight in pg_dict_original.items()]
    weight_list = [weight for volume, weight in pg_dict_original.items()]

    # The action to act
    for key, value in original_response_dict.items():
      match = re.search(r'(\d+\.\d+)', key)
      volume = float(match.group(1))
      lift_weight_list = [float(num) for num in re.findall(r'(\d+\.\d+)', value)]
      # print(lift_weight_list)

      if volume in volume_list:
        pass
      else:
        feedback += f'box[{volume}V] is not in the current warehouse; '
    #except:
    #  feedback = 'Your assigned plan is not in the correct json format as before. If your answer is empty dict, please check whether you miss the left boxes in the warehouse.'

    if feedback != '':
      feedback += 'Please replan for all the agents again with the same ouput format:'
      print('----------Syntactic Check----------')
      print(f'Response original: {response}')
      print(f'Feedback: {feedback}')
      user_prompt_list.append(feedback)
      messages = message_construct_func(user_prompt_list, response_total_list, dialogue_history_method) # message construction
      print(f'Length of messages {len(messages)}')
      response, token_num_count = GPT_response(messages, model_name)
      token_num_count_list_add.append(token_num_count)
      print(f'Response new: {response}\n')
      if response == 'Out of tokens':
        return response, token_num_count_list_add
      iteration_num += 1
    else:
      return response, token_num_count_list_add
  return 'Syntactic Error', token_num_count_list_add


def action_from_response(pg_dict, original_response_dict, lifter_weight_list):
  system_error_feedback = '';
  env_act_feedback = ''
  pg_dict_original = copy.deepcopy(pg_dict)

  # The state to be updated
  volume_list = [volume for volume, weight in pg_dict_original.items()]
  weight_list = [weight for volume, weight in pg_dict_original.items()]

  # The action to act
  for key, value in original_response_dict.items():
    match = re.search(r'(\d+\.\d+)', key)
    volume = float(match.group(1))
    lift_weight_list = [float(num) for num in re.findall(r'(\d+\.\d+)', value)]
    for item in lift_weight_list:
      if item not in lifter_weight_list:
        system_error_feedback += f'agent[{item}W] is not in the current warehouse; '

    if volume in volume_list:
      index = volume_list.index(volume)
      if np.sum(lift_weight_list) >= weight_list[index]:
        volume_list.pop(index)
        weight_list.pop(index)
      else:
        expression = ''
        for index_2 in range(len(lift_weight_list)):
          if index_2 != len(lift_weight_list) - 1:
            expression += f'agent[{lift_weight_list[index_2]}W] and '
          else:
            expression += f'agent[{lift_weight_list[index_2]}W]'
        env_act_feedback += f'The weight of box[{volume}V] is higher than the summation of lifting capability of {expression}, so it can not be lifted. '
    else:
      system_error_feedback += f'box[{volume}V] is not in the current warehouse; '

  pg_dict_original = dict(zip(volume_list, weight_list))
  return system_error_feedback, pg_dict_original, env_act_feedback



def assign_weight(volume):
    # Step 1: Assume a base density to convert volume to weight.
    # This value is an assumption; in real-life, different items have different densities.
    # Let's assume a density of 0.5 kg/m^3 for simplicity. 
    # You can adjust this value based on your requirements.
    density = 1
    estimated_weight = volume * density
    
    # Step 2: Add some randomness to the weight.
    # This can be a combination of gaussian noise and outlier noise.
    noise = random.gauss(0, estimated_weight * 0.1)  # 10% of weight as gaussian noise
    outlier_chance = 0.05  # 5% chance to be an outlier
    if random.random() < outlier_chance:
        noise += random.choice([-1, 1]) * estimated_weight * 0.5  # 50% of weight as outlier noise
    
    weight = max(0.1, estimated_weight + noise)  # ensure weight is not negative
    return weight

def env_create(lifter_num, box_num):
    # Create the volume and weight lists
    volume_list = [random.randint(2, 20)/2 for _ in range(box_num)]
    weight_list = [round(assign_weight(volume), 1) for volume in volume_list]

    # Create the lifter list
    lifter_weight_list = [random.randint(1, 15) / 2 for _ in range(lifter_num)]
    while np.sum(lifter_weight_list) < np.max(weight_list):
        lifter_weight_list = [item + 0.5 for item in lifter_weight_list]

    print('lifter_weight_list: ', lifter_weight_list)
    print('volume_list: ', volume_list)
    print('weight_list: ', weight_list)
    print('Deviation ratio: ', [weight_list[i] / volume_list[i] for i in range(len(volume_list))])
    print('\n')
    return lifter_weight_list, volume_list, weight_list

def create_env3(Saving_path, repeat_num = 4):
  if not os.path.exists(Saving_path):
    os.makedirs(Saving_path, exist_ok=True)
  else:
    shutil.rmtree(Saving_path)
    os.makedirs(Saving_path, exist_ok=True)

  for i, box_num in [(4,10), (6,14), (8,18), (10,24)]:
    if not os.path.exists(Saving_path+f'/env_pg_state_{i}'):
      os.makedirs(Saving_path+f'/env_pg_state_{i}', exist_ok=True)
    else:
      shutil.rmtree(Saving_path+f'/env_pg_state_{i}')
      os.makedirs(Saving_path+f'/env_pg_state_{i}', exist_ok=True)

    for iteration_num in range(repeat_num):
      lifter_weight_list, volume_list, weight_list = env_create(i, box_num)
      os.makedirs(Saving_path+f'/env_pg_state_{i}/pg_state{iteration_num}', exist_ok=True)
      with open(Saving_path+f'/env_pg_state_{i}/pg_state{iteration_num}/lifter_weight_list{iteration_num}.txt', 'w') as f:
        for number in lifter_weight_list:
            f.write(str(number) + '\n')

      with open(Saving_path+f'/env_pg_state_{i}/pg_state{iteration_num}/volume_list{iteration_num}.txt', 'w') as f:
        for number in volume_list:
            f.write(str(number) + '\n')

      with open(Saving_path+f'/env_pg_state_{i}/pg_state{iteration_num}/weight_list{iteration_num}.txt', 'w') as f:
        for number in weight_list:
            f.write(str(number) + '\n')

Code_dir_path = 'path_to_multi-agent-framework/multi-agent-framework/' # Put the current code directory path here
Saving_path = Code_dir_path + 'Env3_BoxLift'
create_env3(Saving_path, repeat_num = 10)

