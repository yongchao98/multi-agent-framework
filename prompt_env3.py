from LLM import *
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")
input_prompt_token_limit = 4000

extra_prompt = 'Each lifting agent can be used only once in each step! You can combine multiple agents to lift one box like "box[3.0V]":"agent[1.5W], agent[2.5W]"! Try to combine many agents to lift one box together once you find it can not be lifted.'

def LLM_summarize_func(state_action_prompt_next_initial):
  prompt1 = f"Please summarize the following content as concise as possible: \n{state_action_prompt_next_initial}"
  messages = [{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": prompt1}]
  response = GPT_response(messages, model_name='gpt-4')
  return response


def input_prompt_1_func(state_update_prompt):
  user_prompt_1 = f'''
  You are a central planner directing lifting agents in a warehouse to lift boxes. Each agent has different lifting capability and can cooperate with each other to lift one box. In summation of lifting capability, the agents can lift all boxes. 
  
  The boxes are identified by their volume, e.g., box[1.4V]. The agents are identified by their lifting weight capability, e.g., agent[1.5W]. Actions are like: "box[1.7V]":"agent[2.5W]", "box[6.0V]":"agent[1.5W], agent[2.5W]".

  Your task is to divide the group of each agent to lift all the boxes. After each step, environments provide updates for the left boxes. Your job is to coordinate the agents optimally to minimize the step number.
  
  Note that the agents can only lift one box at a time. {extra_prompt} [The volume of the box is roughly proportional to the weight of the box, but with some randomness. Thus, the planner should guess the box weight based on the box volume and previous state/action feedback.]
  
  The current left boxes and agents are:
  {state_update_prompt}

  Specify your action plan in this format: {{"box[1.7V]":"agent[1.5W]", "box[3.0V]":"agent[1.5W], agent[2.5W]"}}. Include a box only if it has lifting agents to lift it next. Now, plan the next step:
  '''
  return user_prompt_1


def input_prompt_1_func_total(state_update_prompt, response_total_list,
                                                pg_state_list, dialogue_history_list, env_act_feedback_list,
                                                dialogue_history_method, cen_decen_framework):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(env_act_feedback_list) != 1:
    raise error('state and env act feedback list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1 and cen_decen_framework != 'CMAS':
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
  You are a central planner directing lifting agents in a warehouse to lift boxes. Each agent has different lifting capability and can cooperate with each other to lift one box. In summation of lifting capability, the agents can lift all boxes. 
  
  The boxes are identified by their volume, e.g., box[1.4V]. The agents are identified by their lifting weight capability, e.g., agent[1.5W]. Actions are like: "box[1.7V]":"agent[2.5W]", "box[6.0V]":"agent[1.5W], agent[2.5W]".

  Your task is to divide the group of each agent to lift all the boxes. After each step, environments provide updates for the left boxes. Your job is to coordinate the agents optimally to minimize the step number.
  
  The previous state and action pairs at each step are:
  
  Note that the agents can only lift one box at a time. {extra_prompt} [The volume of the box is roughly proportional to the weight of the box, but with some randomness. Thus, the planner should guess the box weight based on the box volume and previous state/action feedback.]

  The current left boxes and agents are:
  {state_update_prompt}

  Specify your action plan in this format: {{"box[1.7V]":"agent[1.5W]", "box[3.0V]":"agent[1.5W], agent[2.5W], agent[5.5W]"}}. Include a box only if it has lifting agents to lift it next. Now, plan the next step:
    '''
  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history' and cen_decen_framework == 'CMAS':
    pass
  elif dialogue_history_method in (
  '_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
    if dialogue_history_method == '_w_only_state_action_history':
      #print('fdsfdsafadsas')
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\nEnvironment Feedback{i + 1}: {env_act_feedback_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\nEnvironment Feedback{i + 1}: {env_act_feedback_list[i]}\n\n' + state_action_prompt
        #state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\nEnvironment Feedback{i + 1}: {env_act_feedback_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
  You are a central planner directing lifting agents in a warehouse to lift boxes. Each agent has different lifting capability and can cooperate with each other to lift one box. In summation of lifting capability, the agents can lift all boxes. 
  
  The boxes are identified by their volume, e.g., box[1.4V]. The agents are identified by their lifting weight capability, e.g., agent[1.5W]. Actions are like: "box[1.7V]":"agent[2.5W]", "box[6.0V]":"agent[1.5W], agent[2.5W]".

  Your task is to divide the group of each agent to lift all the boxes. After each step, environments provide updates for the left boxes. Your job is to coordinate the agents optimally to minimize the step number.
  
  The previous state and action pairs at each step are:
  {state_action_prompt}
  
  Note that the agents can only lift one box at a time. {extra_prompt} [The volume of the box is roughly proportional to the weight of the box, but with some randomness. Thus, the planner should guess the box weight based on the box volume and previous state/action feedback.]

  The current left boxes and agents are:
  {state_update_prompt}

  Specify your action plan in this format: {{"box[1.7V]":"agent[1.5W]", "box[3.0V]":"agent[1.5W], agent[2.5W]"}}. Include a box only if it has lifting agents to lift it next. Now, plan the next step:
      '''
  #print(f'state_action_prompt: {state_action_prompt}')
  return user_prompt_1

def input_prompt_local_agent_DMAS_dialogue_func(state_update_prompt_local_agent, state_update_prompt_other_agent, dialogue_history, response_total_list,
                                                                         pg_state_list, dialogue_history_list,
                                                                         dialogue_history_method):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1:
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
  You are a box-lifting agent in a warehouse to lift boxes. Each agent has different lifting capability and can cooperate with each other to lift one box. In summation of lifting capability, the agents can lift all boxes. 

  The boxes are identified by their volume, e.g., box[1.4V]. The agents are identified by their lifting weight capability, e.g., agent[1.5W]. Actions are like: "box[1.7V]":"agent[2.5W]", "box[6.0V]":"agent[1.5W], agent[2.5W]".

  The task of the central planner is to divide the group of each agent to lift all the boxes. After each step, environments provide updates for the left boxes. The goal of the group is to coordinate the agents optimally to minimize the step number.

  The current state of yourself is: f'Agent[{lift_weight_item}W]: has lifting capacity {lift_weight_item}W'

  Note that the agents can only lift one box at a time. {extra_prompt} [The volume of the box is roughly proportional to the weight of the box, but with some randomness. Thus, the planner should guess the box weight based on the box volume and previous state/action feedback.]

  The current left boxes and agents are:
  {state_update_prompt}

  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"box[1.7V]":"agent[1.5W]", "box[3.0V]":"agent[1.5W], agent[2.5W]"}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"box[2.7V]":"agent[1.5W]", "box[3.0V]":"agent[1.5W], agent[2.5W],  agent[2.0W]"}}

  Example#2: 
  EXECUTE
  {{"box[2.7V]":"agent[4.5W]", "box[3.0V]":"agent[1.5W], agent[2.5W],  agent[2.0W]"}}

  The previous state and action pairs at each step are:
  {state_action_prompt}

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  The current state is {pg_state_list[-1]}
  The central planner\'s current action plan is: 

  End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan as soon as possible, must strictly follow [Action Output Instruction]!
  Your response:
  '''
  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history':
    pass
  elif dialogue_history_method in ('_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
    if dialogue_history_method == '_w_only_state_action_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        #state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
  You are a box-lifting agent in a warehouse to lift boxes. Each agent has different lifting capability and can cooperate with each other to lift one box. In summation of lifting capability, the agents can lift all boxes. 

  The boxes are identified by their volume, e.g., box[1.4V]. The agents are identified by their lifting weight capability, e.g., agent[1.5W]. Actions are like: "box[1.7V]":"agent[2.5W]", "box[6.0V]":"agent[1.5W], agent[2.5W]".

  The task of the central planner is to divide the group of each agent to lift all the boxes. After each step, environments provide updates for the left boxes. The goal of the group is to coordinate the agents optimally to minimize the step number.

  The current state of yourself is: f'Agent[{lift_weight_item}W]: has lifting capacity {lift_weight_item}W'

  Note that the agents can only lift one box at a time. {extra_prompt} [The volume of the box is roughly proportional to the weight of the box, but with some randomness. Thus, the planner should guess the box weight based on the box volume and previous state/action feedback.]

  The current left boxes and agents are:
  {state_update_prompt}

  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"box[1.7V]":"agent[1.5W]", "box[3.0V]":"agent[1.5W], agent[2.5W]"}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"box[2.7V]":"agent[1.5W]", "box[3.0V]":"agent[1.5W], agent[2.5W],  agent[2.0W]"}}

  Example#2: 
  EXECUTE
  {{"box[2.7V]":"agent[4.5W]", "box[3.0V]":"agent[1.5W], agent[2.5W],  agent[2.0W]"}}

  The previous state and action pairs at each step are:
  {state_action_prompt}

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  The current state is {pg_state_list[-1]}
  The central planner\'s current action plan is: {{{central_response}}}.

  End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan as soon as possible, must strictly follow [Action Output Instruction]!
  Your response:
  '''
  return user_prompt_1


def input_prompt_local_agent_HMAS1_dialogue_func(lift_weight_item, state_update_prompt, central_response, response_total_list, pg_state_list, dialogue_history_list, env_act_feedback_list, dialogue_history_method):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(env_act_feedback_list) != 1:
    raise error('state and env act feedback list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1:
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
  You are a box-lifting agent in a warehouse to lift boxes. Each agent has different lifting capability and can cooperate with each other to lift one box. In summation of lifting capability, the agents can lift all boxes. 

  The boxes are identified by their volume, e.g., box[1.4V]. The agents are identified by their lifting weight capability, e.g., agent[1.5W]. Actions are like: "box[1.7V]":"agent[2.5W]", "box[6.0V]":"agent[1.5W], agent[2.5W]".

  The task of the central planner is to divide the group of each agent to lift all the boxes. After each step, environments provide updates for the left boxes. The goal of the group is to coordinate the agents optimally to minimize the step number.

  The current state of yourself is: f'Agent[{lift_weight_item}W]: has lifting capacity {lift_weight_item}W'

  Note that the agents can only lift one box at a time. {extra_prompt} [The volume of the box is roughly proportional to the weight of the box, but with some randomness. Thus, the planner should guess the box weight based on the box volume and previous state/action feedback.]

  The current left boxes and agents are:
  {state_update_prompt}

  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"box[1.7V]":"agent[1.5W]", "box[3.0V]":"agent[1.5W], agent[2.5W]"}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"box[2.7V]":"agent[1.5W]", "box[3.0V]":"agent[1.5W], agent[2.5W],  agent[2.0W]"}}

  Example#2: 
  EXECUTE
  {{"box[2.7V]":"agent[4.5W]", "box[3.0V]":"agent[1.5W], agent[2.5W],  agent[2.0W]"}}

  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  The current state is {pg_state_list[-1]}
  The central planner\'s current action plan is: 

  End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan as soon as possible, must strictly follow [Action Output Instruction]!
  Your response:
  '''

  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history' and cen_decen_framework == 'CMAS':
    pass
  elif dialogue_history_method in (
  '_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
    if dialogue_history_method == '_w_only_state_action_history':
      #print('fdsfdsafadsas')
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\nEnvironment Feedback{i + 1}: {env_act_feedback_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\nEnvironment Feedback{i + 1}: {env_act_feedback_list[i]}\n\n' + state_action_prompt
        #state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\nEnvironment Feedback{i + 1}: {env_act_feedback_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
  You are a box-lifting agent in a warehouse to lift boxes. Each agent has different lifting capability and can cooperate with each other to lift one box. In summation of lifting capability, the agents can lift all boxes. 

  The boxes are identified by their volume, e.g., box[1.4V]. The agents are identified by their lifting weight capability, e.g., agent[1.5W]. Actions are like: "box[1.7V]":"agent[2.5W]", "box[6.0V]":"agent[1.5W], agent[2.5W]".

  The task of the central planner is to divide the group of each agent to lift all the boxes. After each step, environments provide updates for the left boxes. The goal of the group is to coordinate the agents optimally to minimize the step number.

  The current state of yourself is: f'Agent[{lift_weight_item}W]: has lifting capacity {lift_weight_item}W'

  Note that the agents can only lift one box at a time. {extra_prompt} [The volume of the box is roughly proportional to the weight of the box, but with some randomness. Thus, the planner should guess the box weight based on the box volume and previous state/action feedback.]

  The current left boxes and agents are:
  {state_update_prompt}

  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"box[1.7V]":"agent[1.5W]", "box[3.0V]":"agent[1.5W], agent[2.5W]"}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"box[2.7V]":"agent[1.5W]", "box[3.0V]":"agent[1.5W], agent[2.5W],  agent[2.0W]"}}

  Example#2: 
  EXECUTE
  {{"box[2.7V]":"agent[4.5W]", "box[3.0V]":"agent[1.5W], agent[2.5W],  agent[2.0W]"}}

  The previous state and action pairs at each step are:
  {state_action_prompt}

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  The current state is {pg_state_list[-1]}
  The central planner\'s current action plan is: {{{central_response}}}.

  End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan as soon as possible, must strictly follow [Action Output Instruction]!
  Your response:
    '''
  return user_prompt_1

def input_prompt_local_agent_HMAS2_dialogue_func(lift_weight_item, state_update_prompt, central_response, response_total_list, pg_state_list, dialogue_history_list, env_act_feedback_list, dialogue_history_method):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(env_act_feedback_list) != 1:
    raise error('state and env act feedback list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1:
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
  You are a box-lifting agent in a warehouse to lift boxes. Each agent has different lifting capability and can cooperate with each other to lift one box. In summation of lifting capability, the agents can lift all boxes. 
  
  The boxes are identified by their volume, e.g., box[1.4V]. The agents are identified by their lifting weight capability, e.g., agent[1.5W]. Actions are like: "box[1.7V]":"agent[2.5W]", "box[6.0V]":"agent[1.5W], agent[2.5W]".

  The task of the central planner is to divide the group of each agent to lift all the boxes. After each step, environments provide updates for the left boxes. The goal of the group is to coordinate the agents optimally to minimize the step number.
  
  The current state of yourself is: f'Agent[{lift_weight_item}W]: has lifting capacity {lift_weight_item}W'
  
  Note that the agents can only lift one box at a time. {extra_prompt} [The volume of the box is roughly proportional to the weight of the box, but with some randomness. Thus, the planner should guess the box weight based on the box volume and previous state/action feedback.]

  The current left boxes and agents are:
  {state_update_prompt}
  
  The previous state and action pairs at each step are:
  
  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.
    
  The current state is {pg_state_list[-1]}
  The central planner\'s current action plan is: {{{central_response}}}.

  If you agree with it, respond 'I Agree', without any extra words. If not, briefly explain your objections to the central planner. Your response:
  '''

  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history' and cen_decen_framework == 'CMAS':
    pass
  elif dialogue_history_method in (
  '_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
    if dialogue_history_method == '_w_only_state_action_history':
      #print('fdsfdsafadsas')
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\nEnvironment Feedback{i + 1}: {env_act_feedback_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\nEnvironment Feedback{i + 1}: {env_act_feedback_list[i]}\n\n' + state_action_prompt
        #state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\nEnvironment Feedback{i + 1}: {env_act_feedback_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
  You are a box-lifting agent in a warehouse to lift boxes. Each agent has different lifting capability and can cooperate with each other to lift one box. In summation of lifting capability, the agents can lift all boxes. 
  
  The boxes are identified by their volume, e.g., box[1.4V]. The agents are identified by their lifting weight capability, e.g., agent[1.5W]. Actions are like: "box[1.7V]":"agent[2.5W]", "box[6.0V]":"agent[1.5W], agent[2.5W]".

  The task of the central planner is to divide the group of each agent to lift all the boxes. After each step, environments provide updates for the left boxes. The goal of the group is to coordinate the agents optimally to minimize the step number.
  
  The current state of yourself is: f'Agent[{lift_weight_item}W]: has lifting capacity {lift_weight_item}W'
  
  Note that the agents can only lift one box at a time. {extra_prompt} [The volume of the box is roughly proportional to the weight of the box, but with some randomness. Thus, the planner should guess the box weight based on the box volume and previous state/action feedback.]

  The current left boxes and agents are:
  {state_update_prompt}
  
  The previous state and action pairs at each step are:
  {state_action_prompt}
  
  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.
    
  The current state is {pg_state_list[-1]}
  The central planner\'s current action plan is: {{{central_response}}}.

  If you agree with it, respond 'I Agree', without any extra words. If not, briefly explain your objections to the central planner. Your response:
    '''
  return user_prompt_1

def message_construct_func(user_prompt_list, response_total_list, dialogue_history_method):
  if f'{dialogue_history_method}' == '_w_all_dialogue_history':
    messages=[{"role": "system", "content": "You are a helpful assistant."}]
    #print('length of user_prompt_list', len(user_prompt_list))
    for i in range(len(user_prompt_list)):
      messages.append({"role": "user", "content": user_prompt_list[i]})
      if i < len(user_prompt_list)-1:
        messages.append({"role": "assistant", "content": response_total_list[i]})
    #print('Length of messages', len(messages))
  elif f'{dialogue_history_method}' in ('_wo_any_dialogue_history', '_w_only_state_action_history'):
    messages=[{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": user_prompt_list[-1]})
    #print('Length of messages', len(messages))
  return messages
