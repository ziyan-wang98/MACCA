import argparse
import numpy as np
# from tqdm import tqdm
from os.path import dirname, abspath
import os
from tensorboard.backend.event_processing import event_accumulator
import numpy as np


global_cql_alpha_list= ['0.5','5.0','20.0','50.0','100.0','200.0','500.0']
level_list = ['medium','expert','_replay','mixed','random']
algo_list = ['cfcql','rawcql','cqlslmin','cqlslsoft','cqlslmax','baseline_icq','baseline_omar','baseline_madtkd','bc']
result={}
for algo in algo_list:
    result[algo]={}
    for level in level_list:
        if 'cql' in algo:
            result[algo][level]={}
            for cql_alpha in global_cql_alpha_list:
                result[algo][level][cql_alpha]=[]
        else:
            result[algo][level]=[]
print(result)

parser = argparse.ArgumentParser()

parser.add_argument('--map_name','-mn', type=str,default='6h_vs_8z')

args = parser.parse_args()
final_index= ['final_test_return_mean','final_test_score_reward_mean'] if 'academy' in args.map_name else ['final_test_return_mean','final_test_battle_won_mean']

file_list=[]
file_name_list=[]
file_dir_list=[]
def gci(filepath):
# walk all files under filepath, including subdirectories
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath,fi)            
        if os.path.isdir(fi_d):
            gci(fi_d)                  
        else:
            file_list.append(os.path.join(filepath,fi_d))
            file_name_list.append(fi)
            file_dir_list.append(filepath)
pwd = os.path.join(dirname(dirname(abspath(__file__))),'results','tb_logs')
# recursively walk all files under the directory
gci(os.path.join(pwd,args.map_name))

for ii in range(len(file_list)):
    if file_name_list[ii][:6]=='events':
        ea = event_accumulator.EventAccumulator(file_list[ii])  # init the EventAccumulator
        ea.Reload()  # load all event data
        # key=ea.scalars.Keys()[0]
        if 'final_test_return_mean' in ea.scalars.Keys():
            for jj in range(len(ea.scalars.Items('final_test_return_mean'))):
                if ea.scalars.Items('final_test_n_episodes')[jj].value>50:
                    result_info = []
                    for index in final_index:
                        if len(ea.scalars.Items(index)) == len(ea.scalars.Items('final_test_return_mean')):
                            result_info.append(ea.scalars.Items(index)[jj].value)
                        else:
                            result_info.append('none')

                    result_info.append(ea.scalars.Items('final_test_n_episodes')[jj].value)
                    break
            for level in level_list:
                if level in file_dir_list[ii]:
                    if level =='medium' and '_replay' in file_dir_list[ii]:
                        continue
                    if 'cql_qmix' in file_dir_list[ii]:
                        algo_id = 0
                        if 'raw' in file_dir_list[ii]:algo_id = 1
                        elif 'slmin' in file_dir_list[ii]:algo_id=2
                        elif 'slsoft' in file_dir_list[ii]:algo_id=3
                        elif 'slmax' in file_dir_list[ii]:algo_id=4
                        if 'global_cql_alpha_' in file_dir_list[ii]:
                            alpha_start_index = file_dir_list[ii].index('global_cql_alpha_')+17
                            alpha_end_index = file_dir_list[ii].find('_',alpha_start_index)
                            alpha = file_dir_list[ii][alpha_start_index:alpha_end_index]
                            if alpha not in global_cql_alpha_list:
                                global_cql_alpha_list.append(alpha)
                            if alpha not in result[algo_list[algo_id]][level].keys():
                                result[algo_list[algo_id]][level][alpha] = []
                        # for alpha in global_cql_alpha_list:
                        #     if 'global_cql_alpha_'+alpha in file_dir_list[ii]:
                                # algo_id = 0
                                # if 'raw' in file_dir_list[ii]:algo_id = 1
                                # elif 'slmin' in file_dir_list[ii]:algo_id=2
                                # elif 'slsoft' in file_dir_list[ii]:algo_id=3
                                # elif 'slmax' in file_dir_list[ii]:algo_id=4
                            result[algo_list[algo_id]][level][alpha].append(result_info+[file_dir_list[ii]])
                    else:
                        for algo in algo_list:
                            if algo in file_dir_list[ii]:
                                result[algo][level].append(result_info+[file_dir_list[ii]])
    
import pandas as pd

# prepare data
# result = np.array(result)
writer = pd.ExcelWriter(os.path.join(pwd,args.map_name+'.xlsx'))  # create the excel writer
for key in result.keys():
    data={'exp':[],'return_mean':[],'battle_won':[],'test_episodes':[],'file_name':[]}
    for level in level_list:
        if 'cql' in key:
            for alpha in global_cql_alpha_list:
                if alpha in result[key][level].keys():
                    for ii in range(len(result[key][level][alpha])):
                        data['exp'].append(level+'_'+alpha)
                        data['return_mean'].append(result[key][level][alpha][ii][0])
                        data['battle_won'].append(result[key][level][alpha][ii][1])
                        data['test_episodes'].append(result[key][level][alpha][ii][2])
                        data['file_name'].append(result[key][level][alpha][ii][3])
        else:
            for ii in range(len(result[key][level])):
                data['exp'].append(level)
                data['return_mean'].append(result[key][level][ii][0])
                data['battle_won'].append(result[key][level][ii][1])
                data['test_episodes'].append(result[key][level][ii][2])
                data['file_name'].append(result[key][level][ii][3])


    data_df = pd.DataFrame(data)   # convert ndarray to DataFrame
    data_df.to_excel(writer,key,float_format='%.3f')  # write data_df to the sheet (float_format controls precision)
writer.save()  # save
