from LLM import *
from prompt_env3 import *
from env3_create import *
from sre_constants import error
import random
import os
import json
import re
import copy
import numpy as np
import shutil
import time

# cen_decen_framework = 'DMAS', 'HMAS-1', 'CMAS', 'HMAS-2'
# dialogue_history_method = '_w_all_dialogue_history', '_wo_any_dialogue_history', '_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_previous_round_history'
def run_exp(Saving_path, pg_row_num, iteration_num, query_time_limit, dialogue_history_method = '_w_all_dialogue_history', cen_decen_framework = 'CMAS', model_name = 'gpt-3'):

  Saving_path_result = Saving_path+f'/env_pg_state_{pg_row_num}/pg_state{iteration_num}/{cen_decen_framework}{dialogue_history_method}_{model_name}'

  # specify the path to your dir for saving the results
  os.makedirs(Saving_path_result, exist_ok=True)
  os.makedirs(Saving_path_result+f'/prompt', exist_ok=True)
  os.makedirs(Saving_path_result+f'/response', exist_ok=True)
  os.makedirs(Saving_path_result+f'/pg_state', exist_ok=True)
  os.makedirs(Saving_path_result + f'/dialogue_history', exist_ok=True)

  with open(Saving_path+f'/env_pg_state_{pg_row_num}/pg_state{iteration_num}/lifter_weight_list{iteration_num}.txt', 'r') as file:
    lifter_weight_list = [float(line.strip()) for line in file.readlines()]
  with open(Saving_path+f'/env_pg_state_{pg_row_num}/pg_state{iteration_num}/volume_list{iteration_num}.txt', 'r') as file:
    volume_list = [float(line.strip()) for line in file.readlines()]
  with open(Saving_path+f'/env_pg_state_{pg_row_num}/pg_state{iteration_num}/weight_list{iteration_num}.txt', 'r') as file:
    weight_list = [float(line.strip()) for line in file.readlines()]

  if len(volume_list) != len(weight_list):
    raise error('The length of volume_list and weight_list are not equal!')
  else:
    pg_dict = dict(zip(volume_list, weight_list))

  user_prompt_list = [] # The record list of all the input prompts
  response_total_list = [] # The record list of all the responses
  pg_state_list = [] # The record list of pg states in varied steps
  env_act_feedback_list = [] # The record list of env act feedbacks
  dialogue_history_list = []
  left_box_list = []
  token_num_count_list = []
  pg_state_list.append(pg_dict)
  with open(Saving_path_result+'/pg_state' + '/pg_state'+str(1)+'.json', 'w') as f:
    json.dump(pg_dict, f)

  ### Start the Game! Query LLM for response
  print(f'query_time_limit: {query_time_limit}')
  for index_query_times in range(query_time_limit): # The upper limit of calling LLMs
    state_update_prompt, left_box = state_update_func(pg_dict, lifter_weight_list)
    left_box_list.append(left_box)
    if cen_decen_framework in ('DMAS'):
      print('--------DMAS method starts--------')
      match = None
      count_round = 0
      dialogue_history = ''
      response = '{}'
      while not match and count_round <= 3:
        count_round += 1
        state_update_prompt_local_agent, state_update_prompt_other_agent = state_update_func_local_agent(
                                                                                                         local_agent_row_i,
                                                                                                         pg_dict)
        user_prompt_1 = input_prompt_local_agent_DMAS_dialogue_func(state_update_prompt_local_agent,
                                                                     state_update_prompt_other_agent,
                                                                     dialogue_history, response_total_list,
                                                                     pg_state_list, dialogue_history_list,
                                                                     dialogue_history_method)
        user_prompt_list.append(user_prompt_1)
        with open(Saving_path_result + '/prompt' + '/user_prompt_' + str(index_query_times + 1), 'w') as f:
          f.write(user_prompt_list[-1])
        messages = message_construct_func([user_prompt_1], [], '_w_all_dialogue_history')
        initial_response, token_num_count = GPT_response(messages, model_name)
        token_num_count_list.append(token_num_count)

        dialogue_history += f'[Agent[{local_agent_row_i}]: {initial_response}]\n\n'
        #print(dialogue_history)
        if re.search(r'EXECUTE', initial_response):
          # Search for the pattern that starts with { and ends with }
          print('EXECUTE!')
          match = re.search(r'{.*}', initial_response, re.DOTALL)
          if match:
            response = match.group()
            response, token_num_count_list_add = with_action_syntactic_check_func(pg_dict, response,
                                                                                  [user_prompt_list[-1]],
                                                                                  [],
                                                                                  model_name,
                                                                                  '_w_all_dialogue_history')
            token_num_count_list = token_num_count_list + token_num_count_list_add
            print(f'response: {response}')
            #print(f'User prompt: {user_prompt_1}\n\n')
          break
          break
      dialogue_history_list.append(dialogue_history)
    else:
      if cen_decen_framework in ('CMAS', 'HMAS-1', 'HMAS-1-fast', 'HMAS-2'):
        user_prompt_1 = input_prompt_1_func_total(state_update_prompt, response_total_list,
                                  left_box_list, dialogue_history_list, env_act_feedback_list,
                                  dialogue_history_method, cen_decen_framework)
        user_prompt_list.append(user_prompt_1)
        #print('user_prompt_1: ', user_prompt_1)
        messages = message_construct_func([user_prompt_1], [], '_w_all_dialogue_history') # message construction

      with open(Saving_path_result+'/prompt' + '/user_prompt_'+str(index_query_times+1), 'w') as f:
        f.write(user_prompt_list[-1])
      initial_response, token_num_count = GPT_response(messages, model_name)
      print('Initial response: ', initial_response)
      token_num_count_list.append(token_num_count)
      match = re.search(r'{.*}', initial_response, re.DOTALL)
      if match:
        response = match.group()
        if response[0] == '{' and response[-1] == '}':
          if '{' in response[1:-1] and '}' in response[1:-1]:
            match = re.search(r'{.*}', response[:-1], re.DOTALL)
            if match:
              response = match.group()
          print(f'response: {response}')
          print('----------------Start syntactic check--------------')
          response, token_num_count_list_add = with_action_syntactic_check_func(pg_dict, response, [user_prompt_1], [], model_name, '_w_all_dialogue_history')
          token_num_count_list = token_num_count_list + token_num_count_list_add
          print(f'response: {response}')
        else:
          raise ValueError(f'Response format error: {response}')

      if response == 'Out of tokens':
        success_failure = 'failure over token length limit'
        return user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result
      elif response == 'Syntactic Error':
        success_failure = 'Syntactic Error'
        return user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result

      # Local agent response for checking the feasibility of actions
      if cen_decen_framework == 'HMAS-2':
        print('--------HMAS-2 method starts--------')
        break_mark = False; count_round_HMAS2 = 0

        while break_mark == False and count_round_HMAS2 < 3:
          count_round_HMAS2 += 1
          dialogue_history = f'Central Planner: {response}\n'
          prompt_list_dir = {}; response_list_dir = {}; local_agent_response_list_dir = {}
          local_agent_response_list_dir['feedback1'] = ''

          agent_dict = json.loads(response)
          lift_weight_list_total = []
          for key, value in agent_dict.items():
            lift_weight_list_total += [float(num) for num in re.findall(r'(\d+\.\d+)', value)]

          for lift_weight_item in lifter_weight_list:
            if lift_weight_item in lift_weight_list_total:
              prompt_list_dir[f'Agent[{lift_weight_item}W]'] = []
              response_list_dir[f'Agent[{lift_weight_item}W]'] = []

              local_reprompt = input_prompt_local_agent_HMAS2_dialogue_func(lift_weight_item, state_update_prompt, response,
                                                                            response_total_list, pg_state_list,
                                                                            dialogue_history_list,
                                                                            env_act_feedback_list,
                                                                            dialogue_history_method)

              # print(local_reprompt)
              prompt_list_dir[f'Agent[{lift_weight_item}W]'].append(local_reprompt)
              messages = message_construct_func(
                prompt_list_dir[f'Agent[{lift_weight_item}W]'],
                response_list_dir[f'Agent[{lift_weight_item}W]'],
                '_w_all_dialogue_history')
              response_local_agent, token_num_count = GPT_response(messages, model_name)
              token_num_count_list.append(token_num_count)
              print(f'Agent[{lift_weight_item}W] response: {response_local_agent}')
              if not ('I Agree' in response_local_agent or 'I agree' in response_local_agent):
                local_agent_response_list_dir[
                  'feedback1'] += f'Agent[{lift_weight_item}W]: {response_local_agent}\n'  # collect the response from all the local agents
                dialogue_history += f'Agent[{lift_weight_item}W]: {response_local_agent}\n'

            if local_agent_response_list_dir['feedback1'] != '':
              local_agent_response_list_dir['feedback1'] += '\nThis is the feedback from local agents. If you find some errors in your previous plan, try to modify it. Otherwise, output the same plan as before. The output should have the same json format {"box[1.7V]":"agent[1.5W]", "box[3.0V]":"agent[1.5W], agent[2.5W]"}, as above. Do not explain, just directly output json directory. Your response:'
              messages = message_construct_func([user_prompt_list[-1], local_agent_response_list_dir['feedback1']], [response], '_w_all_dialogue_history') # message construction
              response_central_again, token_num_count = GPT_response(messages, model_name)
              token_num_count_list.append(token_num_count)
              match = re.search(r'{.*}', response_central_again, re.DOTALL)
              if match:
                response = match.group()
                response, token_num_count_list_add = with_action_syntactic_check_func(pg_dict, response_central_again, [user_prompt_list[-1], local_agent_response_list_dir['feedback1']], [response], model_name,
                                                          '_w_all_dialogue_history')
                token_num_count_list = token_num_count_list + token_num_count_list_add
                print(f'Modified plan response: {response}')
            else:
              break_mark = True
              pass

        dialogue_history_list.append(dialogue_history)

      elif cen_decen_framework == 'HMAS-1' or cen_decen_framework == 'HMAS-1-fast':
        print('--------HMAS-1 method starts--------')
        count_round = 0
        dialogue_history = f'Central Planner: {response}\n'
        match = None
        while not match and count_round <= 3:
          count_round += 1

          agent_dict = json.loads(response)
          lift_weight_list_total = []
          for key, value in agent_dict.items():
            lift_weight_list_total += [float(num) for num in re.findall(r'(\d+\.\d+)', value)]

          for lift_weight_item in lifter_weight_list:

            if count_round >= 2 and cen_decen_framework == 'HMAS-1-fast':
              user_prompt_1 = input_prompt_local_agent_HMAS1_dialogue_fast_plan_func(state_update_prompt_local_agent,
                                                                           state_update_prompt_other_agent,
                                                                           dialogue_history, response_total_list, pg_state_list,
                                                                           dialogue_history_list, dialogue_history_method,
                                                                           initial_plan=response)
            else:
              #user_prompt_1 = input_prompt_local_agent_HMAS1_dialogue_func(state_update_prompt_local_agent,
              #                                             state_update_prompt_other_agent, dialogue_history,
              #                                             response_total_list, pg_state_list,
              #                                             dialogue_history_list, dialogue_history_method,
              #                                             initial_plan='')

              user_prompt_1 = input_prompt_local_agent_HMAS2_dialogue_func(lift_weight_item, state_update_prompt, response,
                                                                            response_total_list, pg_state_list,
                                                                            dialogue_history_list,
                                                                            env_act_feedback_list,
                                                                            dialogue_history_method)


            user_prompt_list.append(user_prompt_1)
            with open(Saving_path_result + '/prompt' + '/user_prompt_' + str(index_query_times + 1), 'w') as f:
              f.write(user_prompt_list[-1])
            messages = message_construct_func([user_prompt_1], [], '_w_all_dialogue_history')
            initial_response, token_num_count = GPT_response(messages,
                                            model_name)
            token_num_count_list.append(token_num_count)

            #print('-----------prompt------------\n' + initial_response)
            dialogue_history += f'agent[{lift_weight_item}W]: {initial_response}\n'
            #print(dialogue_history)
            match = re.search(r'{.*}', initial_response, re.DOTALL)
            if match and re.search(r'EXECUTE', initial_response):
              response = match.group()
              response, token_num_count_list_add = with_action_syntactic_check_func(pg_dict, response,
                                                                                    [user_prompt_list[-1]],
                                                                                    [],
                                                                                    model_name,
                                                                                    '_w_all_dialogue_history')
              token_num_count_list = token_num_count_list + token_num_count_list_add
              print(f'response: {response}')
              break
              break
        dialogue_history_list.append(dialogue_history)

    response_total_list.append(response)
    if response == 'Out of tokens':
      success_failure = 'failure over token length limit'
      return user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result
    elif response == 'Syntactic Error':
      success_failure = 'Syntactic Error'
      return user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result

    data = json.loads(response)
    
    with open(Saving_path_result+'/response' + '/response'+str(index_query_times+1)+'.json', 'w') as f:
        json.dump(data, f)
    original_response_dict = json.loads(response_total_list[index_query_times])
    print(pg_dict)
    if cen_decen_framework in ('DMAS', 'HMAS-1', 'HMAS-1-fast'):
      with open(Saving_path_result+'/dialogue_history' + '/dialogue_history'+str(index_query_times)+'.txt', 'w') as f:
          f.write(dialogue_history_list[index_query_times])
    try:
      system_error_feedback, pg_dict_returned, env_act_feedback = action_from_response(pg_dict, original_response_dict, lifter_weight_list)
      env_act_feedback_list.append(env_act_feedback)
      if system_error_feedback != '':
        print('system_error_feedback: ', system_error_feedback)
      if env_act_feedback != '':
        print('env_act_feedback: ', env_act_feedback)
      pg_dict = pg_dict_returned

    except:
      print('Hallucination response: ', response)
      success_failure = 'Hallucination of wrong plan'
      return user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result
    pg_state_list.append(pg_dict)

    with open(Saving_path_result + '/pg_state' + '/pg_state' + str(index_query_times+2) + '.json', 'w') as f:
        json.dump(pg_dict, f)

    # Check whether the task has been completed
    if len(pg_dict) == 0:
      break

  if index_query_times < query_time_limit - 1:
    success_failure = 'success'
  else:
    success_failure = 'failure over query time limit'
  return user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result


Code_dir_path = 'path_to_multi-agent-framework/multi-agent-framework/' # Put the current code directory path here
Saving_path = Code_dir_path + 'Env3_BoxLift'
model_name = 'gpt-4-0613'  #'gpt-4-0613', 'gpt-3.5-turbo-16k-0613'
print(f'-------------------Model name: {model_name}-------------------')

for pg_row_num in [4,6,8,10]:
  if pg_row_num == 8:
    query_time_limit = 25
  else:
    query_time_limit = 20
  for iteration_num in range(10):
    print('-------###-------###-------###-------')
    print(f'Row num is: {pg_row_num}, Iteration num is: {iteration_num}\n\n')

    user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result = run_exp(Saving_path, pg_row_num, iteration_num, query_time_limit, dialogue_history_method='_w_only_state_action_history',
            cen_decen_framework='HMAS-2', model_name = model_name)
    with open(Saving_path_result + '/token_num_count.txt', 'w') as f:
      for token_num_num_count in token_num_count_list:
        f.write(str(token_num_num_count) + '\n')

    with open(Saving_path_result + '/success_failure.txt', 'w') as f:
      f.write(success_failure)

    with open(Saving_path_result + '/env_action_times.txt', 'w') as f:
      f.write(f'{index_query_times+1}')
    print(success_failure)
    print(f'Iteration number: {index_query_times+1}')
