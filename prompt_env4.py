from LLM import *
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")
input_prompt_token_limit = 4000

extra_prompt = 'Each agent can do the actions like: "pick box_1.5_1.0", "move left", "move right", "move to track_1", "move to target".\n' \
               'Note that the agent can only pick the box near its location, their row locations should have difference of 0.5, and column difference should be 0.0, e.g., agent0 is in track_1 and column_3 and can do "pick box_1.5_3.0" or "pick box_0.5_3.0".\n' \
               'The warehouse playground has left side column 0 and right side, if the agent column is at these two sides, they can only move right or move left but not both directions.\n' \
               'If the agent in the target, it can move to the left side of all the tracks\n' \
               'If the agent is in the left side of the track, it can move to the target and drop the box.'

collision_avoidance_prompt = '[Do remember that each position(track and column locations) can only accommodate one agent each step! Hence, you need to avoid the collision with other agents. Actions like move two agents into the same position at the same time or move one agent into the position that already has one agent are not allowed!]'

local_agent_checking_prompt = '[Check whether you will collide with other robots in the next step. Especially avoiding the case that you will collide onto other agents in the next step. Avoid the case that one agent and another agent will move into the same position at the same time. Think step by step about your surrounding agents and whether you will collide with them in the next step.]'

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


def input_prompt_1_func_total(state_update_prompt, response_total_list, system_error_feedback_list,
                                                pg_state_list, dialogue_history_list,
                                                dialogue_history_method, cen_decen_framework, track_row_num, column_num):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(system_error_feedback_list) != 1:
    raise error('state and system_error_feedback_list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1 and cen_decen_framework != 'CMAS':
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
  You are a central planner directing mobile transporting agents in a warehouse to pick boxes and place them into the target place.
  
  Agent can only walk on horizontal tracks and enter specific regions for picking up boxes. Each agent can only hold one box each time.
  There are in total {track_row_num} tracks and {column_num} columns in the warehouse field.
  {extra_prompt}

  Your task is to assign each agent the task in the next step. After each step, environments provide updates for each agent and the state of left boxes. Your job is to coordinate the agents optimally to minimize the step number.
  {collision_avoidance_prompt}
  
  The previous state and action pairs at each step are:

  The current left boxes and agents are:
  {state_update_prompt}

  Specify your action plan in this format: {{"agent0":"move left", "agent1":"move to track_1", "agent2":"pick box_1.5_1.0", "agent3":"move to target", "agent4":"move right", "agent5":"pick box_1.5_3.0"}}. Include an agent only if it has actions in the next step. Now, plan the next step:
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
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\nEnvironment Feedback{i + 1}: {system_error_feedback_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\nEnvironment Feedback{i + 1}: {system_error_feedback_list[i]}\n\n' + state_action_prompt
        #state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\nEnvironment Feedback{i + 1}: {system_error_feedback_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
  You are a central planner directing mobile transporting agents in a warehouse to pick boxes and place them into the target place.
  
  Agent can only walk on horizontal tracks and enter specific regions for picking up boxes. Each agent can only hold one box each time.
  There are in total {track_row_num} tracks and {column_num} columns in the warehouse field.
  {extra_prompt}

  Your task is to assign each agent the task in the next step. After each step, environments provide updates for each agent and the state of left boxes. Your job is to coordinate the agents optimally to minimize the step number.
  {collision_avoidance_prompt}
  
  The previous state and action pairs at each step are:
  {state_action_prompt}

  The current left boxes and agents are:
  {state_update_prompt}

  Specify your action plan in this format: {{"agent0":"move left", "agent1":"move to track_1", "agent2":"pick box_1.5_1.0", "agent3":"move to target", "agent4":"move right", "agent5":"pick box_1.5_3.0"}}. Include an agent only if it has actions in the next step. Now, plan the next step:
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
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
  All the agents coordinate with others together to come out a plan and achieve the goal: match each box with its color-coded target. {collision_avoidance_prompt}
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.


  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move(box_green, position[0.0, 0.0])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, position[0.0, 2.0])"}}

  The previous dialogue history is: {{{dialogue_history}}}
  Think step-by-step about the task and the previous dialogue history. Carefully check and correct them if they made a mistake.
  Respond very concisely but informatively, and do not repeat what others have said. Discuss with others to come up with the best plan.
  Propose exactly one action for yourself at the **current** round.
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
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
  All the agents coordinate with others together to come out a plan and achieve the goal: match each box with its color-coded target. {collision_avoidance_prompt}
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:
  {state_action_prompt}
  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.


  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move(box_green, position[0.0, 0.0])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, position[0.0, 2.0])"}}

  The previous dialogue history is: {{{dialogue_history}}}
  Think step-by-step about the task and the previous dialogue history. Carefully check and correct them if they made a mistake.
  Respond very concisely but informatively, and do not repeat what others have said. Discuss with others to come up with the best plan.
  Propose exactly one action for yourself at the **current** round.
  End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan as soon as possible, must strictly follow [Action Output Instruction]!
  Your response:
    '''

  return user_prompt_1


def input_prompt_local_agent_HMAS1_dialogue_fast_plan_func(state_update_prompt_local_agent, state_update_prompt_other_agent,
                                                 dialogue_history, response_total_list, pg_state_list, dialogue_history_list,
                                                 dialogue_history_method, initial_plan=''):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1:
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
  A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.


  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move(box_green, position[0.0, 0.0])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, position[0.0, 2.0])"}}

  The initial plan is: {{{initial_plan}}}
  The previous dialogue history is: {{{dialogue_history}}}
  Think step-by-step about the task, initial plan, and the previous dialogue history. Carefully check and correct them if they made a mistake.
  {collision_avoidance_prompt} Avoid the situation that the box you are moving will collide with other boxes in the corner. Avoid the case that two boxes move to the same corner at the same step.
  End your response by outputting the final plan, must strictly follow [Action Output Instruction]!
  Your response:
  '''

  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history':
    pass
  elif dialogue_history_method in (
  '_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
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
        # state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
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
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
  A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:
  {state_action_prompt}
  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.


  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move(box_green, position[0.0, 0.0])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, position[0.0, 2.0])"}}

  The initial plan is: {{{initial_plan}}}
  The previous dialogue history is: {{{dialogue_history}}}
  Think step-by-step about the task, initial plan, and the previous dialogue history. Carefully check and correct them if they made a mistake.
  {collision_avoidance_prompt} Avoid the situation that the box you are moving will collide with other boxes in the corner. Avoid the case that two boxes move to the same corner at the same step.
  End your response by outputting the final plan, must strictly follow [Action Output Instruction]!
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
  The central planner\'s current action plan is: {{{central_response}}}.

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

def input_prompt_local_agent_HMAS2_dialogue_func(state_update_prompt, central_response, response_total_list, pg_state_list, dialogue_history_list, system_error_feedback_list, dialogue_history_method, agent_name, track_row_num, column_num):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(system_error_feedback_list) != 1:
    raise error('state and env act feedback list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1:
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
You are a mobile transporting agent in a warehouse to pick boxes and place them into the target place.
Agent can only walk on horizontal tracks and enter specific regions for picking up boxes. Each agent can only hold one box each time.
There are in total {track_row_num} tracks and {column_num} columns in the warehouse field.
{extra_prompt}

The central planner assigns each agent the task in the next step. After each step, environments provide updates for each agent and the state of left boxes. The group goal is to coordinate the agents optimally to minimize the step number.
{collision_avoidance_prompt}

The current state and possible actions of all agents are: {{{state_update_prompt}}}.

The previous state and action pairs at each step are:


Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.
  
The current state is {pg_state_list[-1]}
The central planner\'s current action plan is: {{{central_response}}}.

[You are agent {agent_name}].
{local_agent_checking_prompt}
Think step by step, first analyse for each acting agent the collision risks and then: if you agree with it, respond 'I Agree', without any extra words; if not, briefly explain your objections to the central planner. Your response:
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
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\nEnvironment Feedback{i + 1}: {system_error_feedback_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\nEnvironment Feedback{i + 1}: {system_error_feedback_list[i]}\n\n' + state_action_prompt
        #state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\nEnvironment Feedback{i + 1}: {system_error_feedback_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
You are a mobile transporting agent in a warehouse to pick boxes and place them into the target place.
Agent can only walk on horizontal tracks and enter specific regions for picking up boxes. Each agent can only hold one box each time.
There are in total {track_row_num} tracks and {column_num} columns in the warehouse field.
{extra_prompt}

The central planner assigns each agent the task in the next step. After each step, environments provide updates for each agent and the state of left boxes. The group goal is to coordinate the agents optimally to minimize the step number.
{collision_avoidance_prompt}

The current state and possible actions of all agents are: {{{state_update_prompt}}}.

The previous state and action pairs at each step are:
{state_action_prompt}

Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.
  
The current state is {pg_state_list[-1]}
The central planner\'s current action plan is: {{{central_response}}}.

[You are agent {agent_name}].
{local_agent_checking_prompt}
Think step by step, first analyse for each acting agent the collision risks and then: if you agree with it, respond 'I Agree', without any extra words; if not, briefly explain your objections to the central planner. Your response:
'''
  return user_prompt_1


def input_reprompt_func(state_update_prompt):
  user_reprompt = f'''
  Finished! The updated state is as follows(combined targets and boxes with the same color have been removed):

  {state_update_prompt}

  The output should be like json format like: {{Agent[0.5, 0.5]:move(box_blue, position[0.0, 1.0]), Agent[1.5, 0.5]:move...}}. If no action for one agent in the next step, just do not include its action in the output. Also remember at most one action for each agent in each step. {collision_avoidance_prompt}

  Next step output:
  '''
  return user_reprompt

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
