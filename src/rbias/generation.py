import os
import random
import string
import time

from tqdm import tqdm
import numpy as np
import pandas as pd

from .constants import (VAR_TYPE, C_LADDERS, TASKS, LABEL, PROMPT,
                        QUESTION, NO_COT_INPUT_PROMPT)
from .utils import load_data, load_template

PROMPT_TEMPLATE_PATH = 'experiments/templates/%s/prompts_%s_cause_%d.csv'
DATA_PATH_TEMPLATE = 'data/tubingen/tubingen_%s_data_%d.csv'
DEST_PROMPT_FILE_TEMPLATE = 'prompts_%s_%s_cause_%d.csv'
# change 
PROMPT_TEMPLATE_PATH = 'experiments/templates/%s/prompts_%s_change_%d.csv'
DATA_PATH_TEMPLATE = 'data/tubingen/tubingen_%s_data_%d.csv'
DEST_PROMPT_FILE_TEMPLATE = 'prompts_%s_%s_change_%d.csv'

def gen_rchars(min_length, max_length, num, is_upper = True, 
               is_uniform = True):
    
    if  not is_uniform:
        letters = string.ascii_letters
    else:
        letters = (string.ascii_uppercase if is_upper 
                        else string.ascii_lowercase)
    
    length = random.randrange(min_length, max_length + 1)
    variables = set()
    while len(variables) < num:
        if not is_uniform:
            length = random.randrange(min_length, max_length + 1)
        var = ''.join(random.choice(letters) for i in range(length))
        variables.add(var)
    
    return list(variables)

def gen_variables(size, num_vars, var_type, has_values = False, var_data = None):

    if var_type == VAR_TYPE.TUBINGEN:
        if var_data is None:
            raise ValueError("Invalid input: Variable data file path not provided.")
        vars = list(var_data.sample(size, replace=True).itertuples(
                                        index = False, name = None))
    else:
        m_len, M_len = (1, 1) if var_type == VAR_TYPE.RAND_CHAR else (3, 5)
        if has_values:
            f = lambda x: x + list(map(str.lower, x)) # append values
        else:
            f = lambda x: x # do nothing
        vars = [ f(gen_rchars(m_len, M_len, num_vars)) for i in range(size) ]

    return vars

def substitute_vars(text, vars, var_map):
    res = text
    for t, v in zip(var_map, vars):
        res = res.replace(t, v)
    
    return res


def gen_no_cot_prompts(t_prompts: pd.DataFrame, vars, var_map):
    input_prompts = []
    prompt_labels = []
    for i in tqdm(range(t_prompts.shape[0]), desc = "Generating No-COT prompts"):
        prompt_label = t_prompts[LABEL][i]
        for var in vars:
            prompt = substitute_vars(t_prompts[PROMPT][i], var, var_map)
            question = substitute_vars(t_prompts[QUESTION][i], var, var_map)
            input_prompt = NO_COT_INPUT_PROMPT % (prompt, question)
            input_prompts.append(input_prompt)
            prompt_labels.append(prompt_label)
    
    prompts = pd.DataFrame({PROMPT : input_prompts, LABEL: prompt_labels})

    return prompts

def generate_prompts(var_type, trials, sample_size, var_map, dest_path):
    
    # prompt var_type
    p_var_type = 'tubingen' if var_type != 'rchar' else var_type
    g_var_type = VAR_TYPE.RAND_CHAR if var_type == 'rchar' else VAR_TYPE.TUBINGEN
    # check if dest path exists and create prompts folder
    if not os.path.isdir(dest_path):
        raise FileNotFoundError("The destination path doesn't exist.")
    prompts_folder = f"prompt_{var_type}_{int(time.time())}"
    prompts_folder = os.path.join(dest_path, prompts_folder)
    os.mkdir(prompts_folder)
    for task_id in range(TASKS):
        src_data_path = (None if var_type == 'rchar' else 
                    DATA_PATH_TEMPLATE % (var_type, task_id))
        var_data = (None if var_type == 'rchar' else 
                    pd.read_csv(src_data_path, sep = ',', index_col=False))
        for c_rung in C_LADDERS:
            print(f"Generating prompts for [var_type: {var_type}, rung: {c_rung}, task_no: {task_id}]...")
            
            dest_prompts_folder = os.path.join(prompts_folder, f'{c_rung}_{task_id}')
            os.mkdir(dest_prompts_folder)
            for trial_no in range(trials):
                print("Trial No:", trial_no)
                src_template_path = PROMPT_TEMPLATE_PATH % (c_rung, c_rung, task_id)
                template_prompts = pd.read_csv(src_template_path, sep = '|', index_col=False)
                dest_prompts_file_path = os.path.join(dest_prompts_folder, DEST_PROMPT_FILE_TEMPLATE % (var_type, c_rung, trial_no))
                vars = gen_variables(sample_size, len(var_map[task_id]) // 2, g_var_type,
                                     has_values=True, var_data=var_data)
                # add <do> operator for sub.
                var_map_ext = []
                if c_rung == 'sub':
                    for i, var_row in enumerate(vars):
                        vars[i] = list(var_row)
                        vars[i].extend(gen_rchars(3, 5, 1, is_upper = False))
                    var_map_ext = [ '<do>' ]

                prompts = gen_no_cot_prompts(template_prompts, vars, var_map[task_id] + var_map_ext)
                #print(f"Saving prompts to path: {dest_prompts_file_path}")
                prompts.to_csv(dest_prompts_file_path,index = False, sep = '|')
    
    print("Done!")

def generate_prompts_v2(var_type, trials, sample_size, var_map, dest_path):
    # prompt var_type
    p_var_type = 'tubingen' if var_type != 'rchar' else var_type
    g_var_type = VAR_TYPE.RAND_CHAR if var_type == 'rchar' else VAR_TYPE.TUBINGEN
    # check if dest path exists and create prompts folder
    if not os.path.isdir(dest_path):
        raise FileNotFoundError("The destination path doesn't exist.")
    prompts_folder = f"prompt_{var_type}_{int(time.time())}"
    prompts_folder = os.path.join(dest_path, prompts_folder)
    os.mkdir(prompts_folder)
    for task_id in range(TASKS):
        src_data_path = (None if var_type == 'rchar' else
                    DATA_PATH_TEMPLATE % (var_type, task_id))
        var_data = (None if var_type == 'rchar' else
                    pd.read_csv(src_data_path, sep = ',', index_col=False))
        vars_cols = [ gen_variables(sample_size, len(var_map[task_id]) // 2, g_var_type,
                    has_values=True, var_data=var_data)
                         for i in range(trials) ]
        for c_rung in C_LADDERS:
            print(f"Generating prompts for [var_type: {var_type}, rung: {c_rung}, task_no: {task_id}]...")
            dest_prompts_folder = os.path.join(prompts_folder, f'{c_rung}_{task_id}')
            os.mkdir(dest_prompts_folder)
            for trial_no in range(trials):
                print("Trial No:", trial_no)
                src_template_path = PROMPT_TEMPLATE_PATH % (c_rung, c_rung, task_id)
                template_prompts = pd.read_csv(src_template_path, sep = '|', index_col=False)
                dest_prompts_file_path = os.path.join(dest_prompts_folder, DEST_PROMPT_FILE_TEMPLATE % (var_type, c_rung, trial_no))
                vars = vars_cols[trial_no]
                # add <do> operator for sub.
                var_map_ext = []
                if c_rung == 'sub':
                    for i, var_row in enumerate(vars):
                        vars[i] = list(var_row)
                        vars[i].extend(gen_rchars(3, 5, 1, is_upper = False))
                    var_map_ext = [ '<do>' ]

                prompts = gen_no_cot_prompts(template_prompts, vars, var_map[task_id] + var_map_ext)
                #print(f"Saving prompts to path: {dest_prompts_file_path}")
                prompts.to_csv(dest_prompts_file_path,index = False, sep = '|')
    print("Done!")