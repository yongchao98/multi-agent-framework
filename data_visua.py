import numpy as np
Code_dir_path = 'path_to_multi-agent-framework/multi-agent-framework/' # Put the current code directory path here
Saving_path = Code_dir_path + 'Env2_BoxNet2'

candidate_list = [('CMAS','_wo_any_dialogue_history'), ('CMAS','_w_only_state_action_history'),
                  ('HMAS-2','_wo_any_dialogue_history'), ('HMAS-2','_w_only_state_action_history'),
                  ('HMAS-2','_w_all_dialogue_history'), ('HMAS-1','_w_only_state_action_history')]

print('-------#####------------#####------------#####--------')
Env_action_time_list_best_total = []; API_query_time_list_best_total = []; token_num_list_best_total = []
for pg_row_num, pg_column_num in [(2, 2), (2, 4), (4, 4), (4,8)]:
    Env_action_time_list = []; API_query_time_list = []; token_num_list = [];
    for iteration_num in range(10):
        Env_action_time_list.append(1e10); API_query_time_list.append(1e10); token_num_list.append(1e10)
        for cen_decen_framework, dialogue_history_method in candidate_list:
            #print(f'Row num: {pg_row_num}, Column num: {pg_column_num}, {cen_decen_framework}{dialogue_history_method}')
                with open(saving_path +f'env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/{cen_decen_framework}{dialogue_history_method}/success_failure.txt', 'r') as file:
                    first_line = file.readline().strip()
                #print(first_line)

                if first_line == 'success':
                    with open(saving_path +f'env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/{cen_decen_framework}{dialogue_history_method}/env_action_times.txt', 'r') as file:
                        numbers = [float(line.strip()) for line in file.readlines()]
                    #print('Environment action times', numbers[0])
                    if numbers[0] < Env_action_time_list[iteration_num]:
                        Env_action_time_list[iteration_num] = numbers[0]

                    with open(saving_path +f'env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/{cen_decen_framework}{dialogue_history_method}/token_num_count.txt', 'r') as file:
                        numbers = [float(line.strip()) for line in file.readlines()]
                    #print('API query times', len(numbers))
                    #print('Consuming of token num', np.sum(numbers))
                    if len(numbers) < API_query_time_list[iteration_num]:
                        API_query_time_list[iteration_num] = len(numbers)
                    if np.sum(numbers) < token_num_list[iteration_num]:
                        token_num_list[iteration_num] = np.sum(numbers)

    print(f'Row num: {pg_row_num}, Column num: {pg_column_num}, Environment action times, {Env_action_time_list}')
    print(f'Row num: {pg_row_num}, Column num: {pg_column_num}, API query times, {API_query_time_list}')
    print(f'Row num: {pg_row_num}, Column num: {pg_column_num}, Consuming of token num, {token_num_list}')
    print('\n')
    Env_action_time_list_best_total.append(Env_action_time_list)
    API_query_time_list_best_total.append(API_query_time_list)
    token_num_list_best_total.append(token_num_list)

for cen_decen_framework, dialogue_history_method in candidate_list:
    print('\n')
    success_rate_list = []
    Env_action_time_list = [];
    API_query_time_list = [];
    token_num_list = []
    for index, (pg_row_num, pg_column_num) in enumerate([(2,2), (2,4), (4,4)]):
        print('\n')
        success_rate = 0;
        Env_action_time_cost = 0;
        API_query_time_cost = 0;
        token_num_cost = 0
        success_failure_state_list = []
        for iteration_num in range(4):
                with open(
                        saving_path + f'env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/{cen_decen_framework}{dialogue_history_method}/success_failure.txt',
                        'r') as file:
                    first_line = file.readline().strip()
                    success_failure_state_list.append(first_line)

                if first_line == 'success':
                    success_rate += 1
                    with open(
                            saving_path + f'env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/{cen_decen_framework}{dialogue_history_method}/env_action_times.txt',
                            'r') as file:
                        numbers = [float(line.strip()) for line in file.readlines()]
                    # print('Environment action times', numbers[0])
                    if Env_action_time_list_best_total[index][iteration_num] < 1e10:
                        #print(numbers[0]/Env_action_time_list_best_total[index][iteration_num])
                        Env_action_time_cost += numbers[0]/Env_action_time_list_best_total[index][iteration_num]

                    with open(
                            saving_path + f'env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/{cen_decen_framework}{dialogue_history_method}/token_num_count.txt',
                            'r') as file:
                        numbers = [float(line.strip()) for line in file.readlines()]
                    # print('API query times', len(numbers))
                    # print('Consuming of token num', np.sum(numbers))
                    if API_query_time_list_best_total[index][iteration_num] < 1e10:
                        #print(len(numbers)/API_query_time_list_best_total[index][iteration_num])
                        API_query_time_cost += len(numbers)/API_query_time_list_best_total[index][iteration_num]
                    if token_num_list_best_total[index][iteration_num] < 1e10:
                        #print(np.sum(numbers)/token_num_list_best_total[index][iteration_num])
                        token_num_cost += np.sum(numbers)/token_num_list_best_total[index][iteration_num]
        print(f'Row num: {pg_row_num}, Column num: {pg_column_num}, {cen_decen_framework}{dialogue_history_method}')
        print('success_rate', {success_rate/4})
        print(success_failure_state_list)
        success_rate_list.append(success_rate / 4)
        if success_rate > 0:
            print('Env_action_time_cost', {Env_action_time_cost / success_rate})
            Env_action_time_list.append(Env_action_time_cost / success_rate)

            print('API_query_time_cost', {API_query_time_cost / success_rate})
            API_query_time_list.append(API_query_time_cost / success_rate)

            print('token_num_cost', {token_num_cost / success_rate})
            token_num_list.append(token_num_cost / success_rate)

    print('\n')
    print(f'success_rate: {success_rate_list}, {np.sum(success_rate_list)/len(success_rate_list)}')
    print(f'Env_action_time_cost: {Env_action_time_list}, {np.sum(Env_action_time_list) / len(Env_action_time_list)}')
    print(f'API_query_time_cost: {API_query_time_list}, {np.sum(API_query_time_list) / len(API_query_time_list)}')
    print(f'token_num_cost: {token_num_list}, {np.sum(token_num_list) / len(token_num_list)}')