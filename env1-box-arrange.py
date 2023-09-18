from LLM import *
from prompt_env1 import *
from env1_create import *
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
# dialogue_history_method = '_w_all_dialogue_history', '_wo_any_dialogue_history', '_w_only_state_action_history'
def run_exp(Saving_path, pg_row_num, pg_column_num, iteration_num, query_time_limit, dialogue_history_method = '_w_all_dialogue_history', cen_decen_framework = 'CMAS', model_name = 'gpt-3'):

  Saving_path_result = Saving_path+f'/env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/{cen_decen_framework}{dialogue_history_method}_{model_name}'

  # specify the path to your dir for saving the results
  os.makedirs(Saving_path_result, exist_ok=True)
  os.makedirs(Saving_path_result+f'/prompt', exist_ok=True)
  os.makedirs(Saving_path_result+f'/response', exist_ok=True)
  os.makedirs(Saving_path_result+f'/pg_state', exist_ok=True)
  os.makedirs(Saving_path_result + f'/dialogue_history', exist_ok=True)

  with open(Saving_path+f'/env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/pg_state{iteration_num}.json', 'r') as file:
    pg_dict = json.load(file)

  user_prompt_list = [] # The record list of all the input prompts
  response_total_list = [] # The record list of all the responses
  pg_state_list = [] # The record list of apg states in varied steps
  dialogue_history_list = []
  token_num_count_list = []
  pg_state_list.append(pg_dict)
  with open(Saving_path_result+'/pg_state' + '/pg_state'+str(1)+'.json', 'w') as f:
      json.dump(pg_dict, f)

  ### Start the Game! Query LLM for response
  print(f'query_time_limit: {query_time_limit}')
  for index_query_times in range(query_time_limit): # The upper limit of calling LLMs
    #print(index_query_times)
    print(pg_dict)
    state_update_prompt = state_update_func(pg_row_num, pg_column_num, pg_dict)
    if cen_decen_framework in ('DMAS'):
      print('--------DMAS method starts--------')
      match = None
      count_round = 0
      dialogue_history = ''
      response = '{}'
      while not match and count_round <= 3:
        count_round += 1
        for local_agent_row_i in range(pg_row_num):
          for local_agent_column_j in range(pg_column_num):
            #if f'Agent[{local_agent_row_i + 0.5}, {local_agent_column_j + 0.5}]' in pg_dict:
              state_update_prompt_local_agent, state_update_prompt_other_agent = state_update_func_local_agent(pg_row_num,
                                                                                                               pg_column_num,
                                                                                                               local_agent_row_i,
                                                                                                               local_agent_column_j,
                                                                                                               pg_dict)
              user_prompt_1 = input_prompt_local_agent_DMAS_dialogue_func(state_update_prompt_local_agent,
                                                                           state_update_prompt_other_agent,
                                                                           dialogue_history, response_total_list,
                                                                           pg_state_list, dialogue_history_list,
                                                                           dialogue_history_method)
              user_prompt_list.append(user_prompt_1)
              print(f'User prompt: {user_prompt_1}\n\n')
              with open(Saving_path_result + '/prompt' + '/user_prompt_' + str(index_query_times + 1), 'w') as f:
                f.write(user_prompt_list[-1])
              messages = message_construct_func([user_prompt_1], [], '_w_all_dialogue_history')
              initial_response, token_num_count = GPT_response(messages,model_name)  # 'gpt-4' or 'gpt-3.5-turbo-0301' or 'gpt-4-32k' or 'gpt-3' or 'gpt-4-0613'
              token_num_count_list.append(token_num_count)

              dialogue_history += f'[Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]: {initial_response}]\n\n'
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
                                                                                        '_w_all_dialogue_history',
                                                                                        cen_decen_framework)
                  token_num_count_list = token_num_count_list + token_num_count_list_add
                  print(f'response: {response}')
                  #print(f'User prompt: {user_prompt_1}\n\n')
                break
                break
      dialogue_history_list.append(dialogue_history)
    else:
      if cen_decen_framework in ('CMAS', 'HMAS-1', 'HMAS-1-fast', 'HMAS-2'):
        user_prompt_1 = input_prompt_1_func_total(state_update_prompt, response_total_list,
                                  pg_state_list, dialogue_history_list,
                                  dialogue_history_method, cen_decen_framework)
        user_prompt_list.append(user_prompt_1)

        messages = message_construct_func([user_prompt_1], [], '_w_all_dialogue_history') # message construction

      with open(Saving_path_result+'/prompt' + '/user_prompt_'+str(index_query_times+1), 'w') as f:
        f.write(user_prompt_list[-1])
      initial_response, token_num_count = GPT_response(messages, model_name) # 'gpt-4' or 'gpt-3.5-turbo-0301' or 'gpt-4-32k' or 'gpt-3' or 'gpt-4-0613'
      print('Initial response: ', initial_response)
      token_num_count_list.append(token_num_count)
      match = re.search(r'{.*}', initial_response, re.DOTALL)
      if match:
        response = match.group()
        if response[0] == '{' and response[-1] == '}':
          response, token_num_count_list_add = with_action_syntactic_check_func(pg_dict, response, [user_prompt_1], [], model_name, '_w_all_dialogue_history', cen_decen_framework)
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
        dialogue_history = f'Central Planner: {response}\n'
        #print(f'Original plan response: {response}')
        prompt_list_dir = {}; response_list_dir = {}; local_agent_response_list_dir = {}
        local_agent_response_list_dir['feedback1'] = ''
        agent_dict = json.loads(response)
        for local_agent_row_i in range(pg_row_num):
          for local_agent_column_j in range(pg_column_num):
            if f'Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]' in agent_dict:
              prompt_list_dir[f'Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]'] = []
              response_list_dir[f'Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]'] = []
              state_update_prompt_local_agent, state_update_prompt_other_agent = state_update_func_local_agent(pg_row_num, pg_column_num, local_agent_row_i, local_agent_column_j, pg_dict)

              local_reprompt = input_prompt_local_agent_HMAS2_dialogue_func(state_update_prompt_local_agent, state_update_prompt_other_agent, response, response_total_list, pg_state_list, dialogue_history_list, dialogue_history_method)
              prompt_list_dir[f'Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]'].append(local_reprompt)
              messages = message_construct_func(prompt_list_dir[f'Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]'], response_list_dir[f'Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]'], '_w_all_dialogue_history')
              response_local_agent, token_num_count = GPT_response(messages, model_name)
              token_num_count_list.append(token_num_count)
              #print(f'Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}] response: {response_local_agent}')
              if response_local_agent != 'I Agree':
                local_agent_response_list_dir['feedback1'] += f'Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]: {response_local_agent}\n' # collect the response from all the local agents
                dialogue_history += f'Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]: {response_local_agent}\n'

        if local_agent_response_list_dir['feedback1'] != '':
          local_agent_response_list_dir['feedback1'] += '\nThis is the feedback from local agents. If you find some errors in your previous plan, try to modify it. Otherwise, output the same plan as before. The output should have the same json format {Agent[0.5, 0.5]:move(box_blue, square[0.5, 1.5]), Agent[1.5, 0.5]:move...}, as above. Do not explain, just directly output json directory. Your response:'
          messages = message_construct_func([user_prompt_list[-1], local_agent_response_list_dir['feedback1']], [response], '_w_all_dialogue_history') # message construction
          response_central_again, token_num_count = GPT_response(messages, model_name)
          token_num_count_list.append(token_num_count)
          match = re.search(r'{.*}', response_central_again, re.DOTALL)
          if match:
            response = match.group()
            response, token_num_count_list_add = with_action_syntactic_check_func(pg_dict, response_central_again, [user_prompt_list[-1], local_agent_response_list_dir['feedback1']], [response], model_name,
                                                      '_w_all_dialogue_history', cen_decen_framework)
            token_num_count_list = token_num_count_list + token_num_count_list_add
            print(f'response: {response}')
          #print(messages[2])
          #print(messages[3])
          print(f'Modified plan response:\n {response}')
        else:
          print(f'Plan:\n {response}')
          pass

        dialogue_history_list.append(dialogue_history)

      elif cen_decen_framework == 'HMAS-1' or cen_decen_framework == 'HMAS-1-fast':
        print('--------HMAS-1 method starts--------')
        count_round = 0
        dialogue_history = f'Central Planner: {response}\n'
        match = None
        agent_dict = json.loads(response)
        while not match and count_round <= 3:
          count_round += 1
          for local_agent_row_i in range(pg_row_num):
            for local_agent_column_j in range(pg_column_num):
              if f'Agent[{local_agent_row_i + 0.5}, {local_agent_column_j + 0.5}]' in agent_dict:
                state_update_prompt_local_agent, state_update_prompt_other_agent = state_update_func_local_agent(
                  pg_row_num,
                  pg_column_num,
                  local_agent_row_i,
                  local_agent_column_j,
                  pg_dict)
                if count_round >= 2 and cen_decen_framework == 'HMAS-1-fast':
                  user_prompt_1 = input_prompt_local_agent_HMAS1_dialogue_fast_plan_func(state_update_prompt_local_agent,
                                                                               state_update_prompt_other_agent,
                                                                               dialogue_history, response_total_list, pg_state_list,
                                                                               dialogue_history_list, dialogue_history_method,
                                                                               initial_plan=response)
                else:
                  user_prompt_1 = input_prompt_local_agent_HMAS1_dialogue_func(state_update_prompt_local_agent,
                                                               state_update_prompt_other_agent, dialogue_history,
                                                               response_total_list, pg_state_list,
                                                               dialogue_history_list, dialogue_history_method,
                                                               initial_plan='')

                user_prompt_list.append(user_prompt_1)
                with open(Saving_path_result + '/prompt' + '/user_prompt_' + str(index_query_times + 1), 'w') as f:
                  f.write(user_prompt_list[-1])
                messages = message_construct_func([user_prompt_1], [], '_w_all_dialogue_history')
                initial_response, token_num_count = GPT_response(messages,model_name)  # 'gpt-4' or 'gpt-3.5-turbo-0301' or 'gpt-4-32k' or 'gpt-3' or 'gpt-4-0613'
                token_num_count_list.append(token_num_count)

                #print('-----------prompt------------\n' + initial_response)
                dialogue_history += f'Agent[{local_agent_row_i + 0.5}, {local_agent_column_j + 0.5}]: {initial_response}\n'
                #print(dialogue_history)
                match = re.search(r'{.*}', initial_response, re.DOTALL)
                if match and re.search(r'EXECUTE', initial_response):
                  response = match.group()
                  response, token_num_count_list_add = with_action_syntactic_check_func(pg_dict, response,
                                                                                        [user_prompt_list[-1]],
                                                                                        [],
                                                                                        model_name,
                                                                                        '_w_all_dialogue_history',
                                                                                        cen_decen_framework)
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
      system_error_feedback, pg_dict_returned = action_from_response(pg_dict, original_response_dict)
      if system_error_feedback != '':
        print(system_error_feedback)
      pg_dict = pg_dict_returned

    except:
      success_failure = 'Hallucination of wrong plan'
      return user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result
    pg_state_list.append(pg_dict)
    with open(Saving_path_result+'/pg_state' + '/pg_state'+str(index_query_times+2)+'.json', 'w') as f:
        json.dump(pg_dict, f)

    # Check whether the task has been completed
    count = 0
    for key, value in pg_dict.items():
      count += len(value)
    if count == 0:
      break

  if index_query_times < query_time_limit - 1:
    success_failure = 'success'
  else:
    success_failure = 'failure over query time limit'
  return user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result


Code_dir_path = 'path_to_multi-agent-framework/multi-agent-framework/' # Put the current code directory path here
Saving_path = Code_dir_path + 'Env2_BoxNet1'
model_name = 'gpt-4-0613'  #'gpt-4-0613', 'gpt-3.5-turbo-16k-0613'
print(f'-------------------Model name: {model_name}-------------------')
for pg_row_num, pg_column_num in [(2,2), (2,4), (4,4), (4,8)]:
  if pg_row_num == 4 and pg_column_num == 8:
    query_time_limit = 40
  else:
    query_time_limit = 30
  for iteration_num in range(10):
    print('-------###-------###-------###-------')
    print(f'Row num is: {pg_row_num}, Column num is: {pg_column_num}, Iteration num is: {iteration_num}\n\n')

    user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result = run_exp(Saving_path, pg_row_num, pg_column_num, iteration_num, query_time_limit, dialogue_history_method='_w_only_state_action_history',
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