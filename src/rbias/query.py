import os
import glob
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import metrics
from .constants import (VAR_TYPE, C_LADDERS, TASKS, LABEL, PROMPT, RESPONSE, PRED, R_PRED)

from .llm import Model, init_llm

def get_llm(model_name): # needs to be modified. a dumpy function for now
    return (lambda x: "yes" if np.random.rand() > 0.5 else "no")

def analyze_result(df: pd.DataFrame): 
    y_true, y_pred, r_pred = df[LABEL], df[PRED], df[R_PRED]
    # to handle `err` values - should result in false
    ny_true = np.where(y_true == 'yes', 'no', 'yes')
    y_pred = np.where(y_pred == 'err', ny_true, y_pred)

    yes_pred = np.repeat('yes', len(y_true))
    no_pred = np.repeat('no', len(y_true))

    acc = (y_true == y_pred).mean()
    r_acc = (y_true == r_pred).mean()
    yes_acc = (y_true == yes_pred).mean()
    no_acc = (y_true == no_pred).mean()
    f1_score = metrics.f1_score(y_true, y_pred, pos_label='yes')
    return acc, r_acc, yes_acc, no_acc, f1_score, y_true.size

def query_llm(llm, prompts) -> pd.DataFrame:
    results, query_log = llm(prompts[PROMPT])
    y_pred, response = tuple(map(list, zip(*results)))
    r_pred = np.random.choice(['yes', 'no'], len(y_pred))
    rdf = pd.DataFrame({PROMPT: prompts[PROMPT], RESPONSE: response, 
                        LABEL: prompts[LABEL], PRED: y_pred, R_PRED: r_pred})
    print("query log:", query_log)
    return rdf, np.array(query_log)

def run_prompts(model_name, prompts_path, results_path, run_id):
    # prompts path check
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"Prompts folder path  doesn't exits: {prompts_path}")
    
    # initialize model
    print(f"Initializing model...{model_name}.")
    llm = init_llm(Model._value2member_map_[model_name])
    
    # result folders and path check
    prompt_id = re.search("prompt_([a-z]+_[0-9]+)", prompts_path).group(1)
    res_folder_name = f"result_{prompt_id}"
    res_folder_path = os.path.join(results_path, res_folder_name)
    if not os.path.isdir(res_folder_path):
        os.mkdir(res_folder_path)
    
    # Create unique run_id folder
    run_folder =  f"{model_name}_{run_id}"
    res_folder_path = os.path.join(res_folder_path, run_folder)
    if os.path.isdir(res_folder_path):
        raise FileExistsError(f"Folder: {res_folder_path} already exits.")
    os.mkdir(res_folder_path)
    print(f"Creating result run folder: {res_folder_name}/{run_folder}")

    results = []
    log_results = []
    for c_rung in C_LADDERS:
        query_log_total = np.array([0, 0, 0])
        for task_id in range(TASKS):
            prompt_files = sorted(glob.glob(f"{prompts_path}/{c_rung}_{task_id}/*.csv"))
            os.mkdir(os.path.join(res_folder_path,f"{c_rung}_{task_id}"))
            print(f"type: {c_rung}, task: {task_id}")
            trials = len(prompt_files)
            for i in tqdm(range(trials), "Trials"):
                pfile = prompt_files[i]
                pdf = pd.read_csv(pfile,  sep = '|', index_col=False)
                response_df, query_log = query_llm(llm, pdf)
                # store the responses in a file
                response_file_path = os.path.join(res_folder_path, f"{c_rung}_{task_id}", 
                                                    f"response_{i}.csv")
                response_df.to_csv(response_file_path, sep = '|')
                # use the result dataframe to generate/tabulate result and other metrics
                acc, r_acc, yes_acc, no_acc, f1_score, size = analyze_result(response_df)
                results.append((f"{c_rung}_{task_id}", i, size, acc, r_acc, yes_acc, 
                                no_acc, f1_score))
                query_log_total += query_log
        log_results.append(query_log_total)

    # store the responses, results and other metrices
    res_df = pd.DataFrame(results, columns = ["exp_name", "trial", "size", "acc", "r_acc", 
                                                "yes_acc", "no_acc", "f1_score"])
    res_file_path = os.path.join(res_folder_path, "results.csv")
    res_df.to_csv(res_file_path, index=True)

    # store query logs
    log_file_path = os.path.join(res_folder_path, "query_log.csv")
    log_df = pd.DataFrame(log_results, columns = ["no_retry", "retry", "failure"])
    log_df.insert(0, 'c_rung', C_LADDERS)
    log_df.to_csv(log_file_path, index=True)

    print("Done!")