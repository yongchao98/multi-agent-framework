from prompt_env4 import *
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

def state_update_func(agent_position_state_dict, box_position_dict, track_row_num, column_num):
  state_update_prompt = f'The states and actions of available agents are: \n'
  state_update_prompt += f'The left boxes and their locations in the warehouse are: '
  for key, value in box_position_dict.items():
    if value == 1:
      state_update_prompt += f'box_{key}, '
  state_update_prompt += f'.\n'

  for i in range(len(agent_position_state_dict)):
    if type(agent_position_state_dict[f'agent{i}']) == str and agent_position_state_dict[f'agent{i}'] == 'target':
      state_update_prompt += f'I am agent{i}, I am in target now, I can do: '
      for row_num in range(track_row_num):
        state_update_prompt += f'move to track_{row_num}; '
    else:
      if agent_position_state_dict[f'agent{i}'][2] == 1:
          state_update_prompt += f'I am agent{i}, I am in track_{agent_position_state_dict[f"agent{i}"][0]} and column_{agent_position_state_dict[f"agent{i}"][1]}, I am having box on myself so can not pick more box now. I can do: '
      else:
          state_update_prompt += f'I am agent{i}, I am in track_{agent_position_state_dict[f"agent{i}"][0]} and column_{agent_position_state_dict[f"agent{i}"][1]}, I am not having box on myself so can pick one box. I can do: '
      if agent_position_state_dict[f'agent{i}'][1] > 0:
        state_update_prompt += f'move left; '
      if agent_position_state_dict[f'agent{i}'][1] < column_num-1:
        state_update_prompt += f'move right; '
      if agent_position_state_dict[f'agent{i}'][1] == 0:
        state_update_prompt += f'move to target; '
      if agent_position_state_dict[f'agent{i}'][0] - 0.5 > 0 and box_position_dict[f'{agent_position_state_dict[f"agent{i}"][0]-0.5}_{float(agent_position_state_dict[f"agent{i}"][1])}'] == 1 and agent_position_state_dict[f'agent{i}'][2] == 0:
        state_update_prompt += f'pick box_{agent_position_state_dict[f"agent{i}"][0]-0.5}_{float(agent_position_state_dict[f"agent{i}"][1])}; '
      if agent_position_state_dict[f'agent{i}'][0] + 0.5 < track_row_num-1 and box_position_dict[f'{agent_position_state_dict[f"agent{i}"][0] + 0.5}_{float(agent_position_state_dict[f"agent{i}"][1])}'] == 1 and agent_position_state_dict[f'agent{i}'][2] == 0:
        state_update_prompt += f'pick box_{agent_position_state_dict[f"agent{i}"][0]+0.5}_{float(agent_position_state_dict[f"agent{i}"][1])}; '
    state_update_prompt += f'.\n'

  state_update_prompt += f'\n'
  return state_update_prompt


def action_from_response(pg_dict_input, original_response_dict, track_row_num, column_num, box_position_dict_input):
  collision_check = False
  system_error_feedback = ''
  pg_dict_original = copy.deepcopy(pg_dict_input)
  box_position_dict = copy.deepcopy(box_position_dict_input)

  for key, value in original_response_dict.items():
    # '{"agent0":"move left", "agent1":"pick box_1.0_1.5"}'

    if 'left' in value:
      if pg_dict_original[key][1]>0:
        pg_dict_original[key][1] -= 1
      elif pg_dict_original[key][1]==0:
        system_error_feedback += f'{key} has arrived at the left side of the track, you can not move left. You can enter_target to leave the track and drop the box (if you have box), or you can move left or pick up near boxes.'
      else:
        print(f"Error, Key: {key}, Value: {value}")
        system_error_feedback += f'Your assigned task for {key} is not in the doable action list; '
    elif 'right' in value:
      if pg_dict_original[key][1]<column_num-1:
        pg_dict_original[key][1] += 1
      elif pg_dict_original[key][1]==column_num-1:
        print(f"Error, Key: {key}, Value: {value}")
        system_error_feedback += f'{key} has arrived at the right side of the track, you can not move right but move left.'
      else:
        print(f"Error, Key: {key}, Value: {value}")
        system_error_feedback += f'Your assigned task for {key} is not in the doable action list; '
    elif 'target' in value:
      if pg_dict_original[key][1] == 0:
        pg_dict_original[key] = 'target'
      else:
        print(f"Error, Key: {key}, Value: {value}")
        system_error_feedback += f'Your assigned task for {key} is not in the doable action list; '
    elif 'pick' in value:
      pattern = r"(\d+\.\d+)"
      numbers = re.findall(pattern, value)
      float_numbers = [float(num) for num in numbers]

      if pg_dict_original[key][2] == 0 and pg_dict_original[key][1] == float_numbers[1] and np.abs(pg_dict_original[key][0] - float_numbers[0])==0.5 and box_position_dict[f'{float_numbers[0]}_{float_numbers[1]}'] == 1:
        pg_dict_original[key][2] = 1
        box_position_dict[f'{float_numbers[0]}_{float_numbers[1]}'] = 0
      elif pg_dict_original[key][2] == 1:
        system_error_feedback += f'{key} already has one box on it.'
      elif box_position_dict[f'{float_numbers[0]}_{float_numbers[1]}'] == 0:
        system_error_feedback += f'Location {float_numbers[0]}_{float_numbers[1]} does not have one box on it.'
      else:
        print(f"Error, Key: {key}, Value: {value}")
        system_error_feedback += f'Your assigned task for {key} is not in the doable action list; '
    elif 'track' in value:
      pattern = r'\d+'
      numbers = re.findall(pattern, value)
      float_numbers = [int(num) for num in numbers]
      if len(float_numbers) != 1 or float_numbers[0] >= track_row_num or float_numbers[0] <= 0:
        print(f"Error, Key: {key}, Value: {value}")
        system_error_feedback += f'Your assigned task for {key} is not in the doable action list; '
      if pg_dict_original[key] == 'target':
        #print(float_numbers)
        pg_dict_original[key] = [float_numbers[0], 0, 0]

    else:
      print(f"Error, Key: {key}, Value: {value}")
      system_error_feedback += f'Your assigned task for {key} is not in the doable action list; '

  position_list = []
  for key, value in pg_dict_original.items():
    if value == 'target':
      pass
    else:
      if [value[0], value[1]] in position_list:
        collision_check = True
        break
      else:
        position_list.append([value[0], value[1]])

  return system_error_feedback, pg_dict_original, collision_check, box_position_dict


def with_action_syntactic_check_func(pg_dict_input, response, user_prompt_list_input, response_total_list_input,
                                     model_name, dialogue_history_method, track_row_num, column_num, box_position_dict):
  #print('----------Syntactic Check----------')
  user_prompt_list = copy.deepcopy(user_prompt_list_input)
  response_total_list = copy.deepcopy(response_total_list_input)
  iteration_num = 0
  token_num_count_list_add = []
  while iteration_num < 6:
    response_total_list.append(response)
    # try:
    original_response_dict = json.loads(response)
    pg_dict_original = copy.deepcopy(pg_dict_input)
    system_error_feedback, pg_dict_original_return, collision_check_return, box_position_dict_return = action_from_response(
      pg_dict_original, original_response_dict, track_row_num, column_num, box_position_dict)
    feedback = system_error_feedback

    if feedback != '':
      feedback += 'Please replan for all the agents again with the same ouput format:'
      print('----------Syntactic Check----------')
      print(f'Response original: {response}')
      print(f'Feedback: {feedback}')
      user_prompt_list.append(feedback)
      messages = message_construct_func(user_prompt_list, response_total_list,
                                        dialogue_history_method)  # message construction
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

def generate_unique_integers(lower_limit, upper_limit, m):
  if m > (upper_limit - lower_limit + 1):
    raise ValueError("Cannot generate more unique integers than the range allows.")
  unique_integers = set()
  while len(unique_integers) < m:
    unique_integers.add(random.randint(lower_limit, upper_limit))
  return list(unique_integers)

def env_create(track_row_num, column_num, box_occupy_ratio, agent_num):
  box_position_dict = {}
  # assign boxes to positions
  for i in range(track_row_num -1):
    for j in range(column_num):
      if random.random() < box_occupy_ratio:
        box_position_dict[f'{float(i+0.5)}_{float(j)}'] = 1
      else:
        box_position_dict[f'{float(i+0.5)}_{float(j)}'] = 0

  # assign agents into positions
  agent_position_state_dict = {}

  lower_limit = 0
  upper_limit = track_row_num * column_num - 1
  unique_integers = generate_unique_integers(lower_limit, upper_limit + 5, agent_num)

  for i in range(agent_num):
    if unique_integers[i] > upper_limit:
      agent_position_state_dict[f'agent{i}'] = 'target'
    else:
      agent_position_state_dict[f'agent{i}'] = (unique_integers[i] // column_num, unique_integers[i] % column_num, 0)
  return agent_position_state_dict, box_position_dict


def create_env4(Saving_path, repeat_num = 10):
  if not os.path.exists(Saving_path):
    os.makedirs(Saving_path, exist_ok=True)
  else:
    shutil.rmtree(Saving_path)
    os.makedirs(Saving_path, exist_ok=True)

  for track_row_num, column_num, box_occupy_ratio, agent_num in [(3, 5, 0.5, 4)]:
    if not os.path.exists(Saving_path+f'/env_pg_state_{track_row_num}_{column_num}_{box_occupy_ratio}_{agent_num}'):
      os.makedirs(Saving_path+f'/env_pg_state_{track_row_num}_{column_num}_{box_occupy_ratio}_{agent_num}', exist_ok=True)
    else:
      shutil.rmtree(Saving_path+f'/env_pg_state_{track_row_num}_{column_num}_{box_occupy_ratio}_{agent_num}')
      os.makedirs(Saving_path+f'/env_pg_state_{track_row_num}_{column_num}_{box_occupy_ratio}_{agent_num}', exist_ok=True)

    for iteration_num in range(repeat_num):
      agent_position_state_dict, box_position_dict = env_create(track_row_num, column_num, box_occupy_ratio, agent_num)
      print('Initial agent state: ', agent_position_state_dict)
      print('Box_matrix: ', box_position_dict)
      os.makedirs(Saving_path+f'/env_pg_state_{track_row_num}_{column_num}_{box_occupy_ratio}_{agent_num}/pg_state{iteration_num}', exist_ok=True)
      with open(Saving_path+f'/env_pg_state_{track_row_num}_{column_num}_{box_occupy_ratio}_{agent_num}/pg_state{iteration_num}/pg_state{iteration_num}.json', 'w') as f:
        json.dump(agent_position_state_dict, f)
      with open(Saving_path+f'/env_pg_state_{track_row_num}_{column_num}_{box_occupy_ratio}_{agent_num}/pg_state{iteration_num}/box_state{iteration_num}.json', 'w') as f:
        json.dump(box_position_dict, f)
      print('\n')

Code_dir_path = 'path_to_multi-agent-framework/multi-agent-framework/' # Put the current code directory path here
Saving_path = Code_dir_path + 'Env4_Warehouse'
model_name = 'gpt-4-0613'  #'gpt-4-0613', 'gpt-3.5-turbo-16k-0613'
# The first time to create the environment, after that you can comment it
create_env4(Saving_path, repeat_num = 10)