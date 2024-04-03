
import os
import glob
import numpy as np
import pandas as pd
from .constants import (DATASETS, C_LADDERS, TASKS, OBS_INTER_MAP, 
                        NUM_INTERS, INTER_VARS)
from .query import analyze_result

def prep_taskwise_across_data_rungs(result_files):
    p_data = []
    for f, d in zip(result_files, DATASETS):
        df = pd.read_csv(f, index_col=0)
        arr = df["acc"].to_numpy()
        arr = arr.reshape((3, 3, -1))
        avg, var = np.mean(arr, axis=2), np.std(arr, axis=2)
        sub_arr = np.concatenate((avg[..., None], var[..., None]), axis = 2)
        p_data.append(sub_arr)
    
    p_data = np.array(p_data)
    res_across_rungs = np.transpose(p_data, axes= [2, 1, 0, 3])
    res_across_data = np.transpose(p_data, axes=[2, 0, 1, 3]) 

    return res_across_rungs, res_across_data


def sep_responses_based_on_inter(response_folders, filters):
    print("Separating responses according to obvious and non-obvious intervenstions...")
    for folder_path in response_folders:
        print("result_folder: ", folder_path)
        obv_res = []
        non_obv_res = []
        for rung in C_LADDERS[1:]:
            for task_id in range(TASKS):
                exp_name = f"{rung}_{task_id}"
                print("Exp name:", exp_name)
                sub_folder_path = os.path.join(folder_path, exp_name)
                files = sorted(glob.glob(f"{sub_folder_path}/*.csv"))
                for i, file in enumerate(files):
                    df = pd.read_csv(file, index_col=0, sep = "|")
                    # separate them into two dfs - one that change and the other that doesn't
                    obv_inter, non_obv_inter = df.loc[filters[task_id][0]], df.loc[filters[task_id][1]]
                    
                    # store them in separate folders
                    obv_path = os.path.join(sub_folder_path, "obv")
                    if not os.path.isdir(obv_path):
                        os.mkdir(obv_path)
                    non_obv_path = os.path.join(sub_folder_path, "non_obv")
                    if not os.path.isdir(non_obv_path):
                        os.mkdir(non_obv_path)
                    obv_inter.to_csv(os.path.join(obv_path, f"response_{i}.csv"), sep="|")
                    non_obv_inter.to_csv(os.path.join(non_obv_path, f"response_{i}.csv"), sep="|")
                    # generate separate results for obv and non_obv
                    g_res = analyze_result(obv_inter)
                    b_res = analyze_result(non_obv_inter)
                    obv_res.append([exp_name, i, g_res[-1], *(g_res[:-1])])
                    non_obv_res.append([exp_name, i, b_res[-1], *(b_res[:-1])])         
        # store the result files
        obv_res_df = pd.DataFrame(obv_res, columns = ["exp_name", "trial","size", "acc", "r_acc", "yes_acc", "no_acc", "f1_score"])
        non_obv_res_df = pd.DataFrame(non_obv_res, columns = ["exp_name", "trial", "size", "acc", "r_acc", "yes_acc", "no_acc", "f1_score"])
        obv_res_df.to_csv(os.path.join(folder_path, "obv_results.csv"))
        non_obv_res_df.to_csv(os.path.join(folder_path, "non_obv_results.csv"))
    print("Done!")
    

def prep_taskwise_obv_non_obv_inter(res_folder_paths):
    obv_data = []
    for f, d in zip(res_folder_paths, DATASETS):
        f = os.path.join(f, "obv_results.csv")
        df = pd.read_csv(f, index_col=0)
        arr = df["acc"].to_numpy()
        arr = arr.reshape((2, TASKS, -1))
        avg, std = np.mean(arr, axis=2), np.std(arr, axis=2)
        sub_arr = np.concatenate((avg[..., None], std[..., None]), axis = 2)
        obv_data.append(sub_arr)

    non_obv_data = []
    for f, d in zip(res_folder_paths, DATASETS):
        f = os.path.join(f, "non_obv_results.csv")
        df = pd.read_csv(f, index_col=0)
        arr = df["acc"].to_numpy()
        arr = arr.reshape((2, TASKS, -1))
        avg, std = np.mean(arr, axis=2), np.std(arr, axis=2)
        sub_arr = np.concatenate((avg[..., None], std[..., None]), axis = 2)
        non_obv_data.append(sub_arr)
    data = [obv_data, non_obv_data]
    # (obv/non) - dataset - (inter/sub) - tasks - (avg/std) -> (inter/sub) - (tasks) - (dataset) - (obv/non) - (avg/std)
    data = np.array(data).transpose([2, 3, 1, 0, 4])
    
    return data

def compare_obs_inter_perf(res_files, sample_size):
    obs_fold_template = "obs_%s"
    inter_fold_template = "inter_%s"
    
    for f in res_files:
        obs_inter_comp_res = []
        obs_inter_comp_out = []
        for task_id in range(TASKS):
            obs_fold = os.path.join(f, obs_fold_template % (task_id), "response_*.csv")
            inter_fold  = os.path.join(f, inter_fold_template % (task_id), "response_*.csv")
            obs_files = sorted(glob.glob(obs_fold))
            inter_files = sorted(glob.glob(inter_fold))
            obs_arrs = []
            inter_arrs = []
            for trial_no, (obs_file, inter_file) in enumerate(zip(obs_files, inter_files)):
                entry = [task_id, ]
                print(f"task_id: {task_id}, trial_no: {trial_no}")
                obs_resp = pd.read_csv(obs_file, sep = "|", index_col=0)
                inter_resp = pd.read_csv(inter_file, sep = "|", index_col=0)
                obs_arr = (obs_resp["label"] == obs_resp["pred"]).astype(int).to_numpy().reshape((sample_size, -1))
                inter_arr =  (inter_resp["label"] == inter_resp["pred"]).astype(int).to_numpy().reshape((sample_size, -1))
                nobs_arr = obs_arr[:, OBS_INTER_MAP[task_id]]
                obs_arrs.append(nobs_arr)
                inter_arrs.append(inter_arr)
            obs_arr = np.concatenate(obs_arrs, axis=0)
            inter_arr = np.concatenate(inter_arrs, axis=0)
            #print("obs_inter", obs_arr.shape, inter_arr.shape, obs_arr, inter_arr)
            res = (inter_arr & obs_arr).reshape(15, NUM_INTERS[task_id], -1).swapaxes(0,1)
            res_acc = res.mean(axis=-1)
            res_m, res_std = np.mean(res_acc, axis=-1), np.std(res_acc, axis=-1)
            #print(res.shape, res_acc.shape, res)
            trials = res.shape[1]
            for i, inter_var in enumerate(INTER_VARS[task_id]):
                for trial_no in range(trials):
                    obs_inter_comp_out.append([task_id, inter_var, trial_no, res[i][trial_no], res_acc[i][trial_no]])
                obs_inter_comp_res.append([task_id, inter_var, trials, res_m[i], res_std[i]])
        inter_obs_comp_out_file = os.path.join(f, "obs_inter_output.csv")
        inter_obs_comp_res_file = os.path.join(f, "obs_inter_results.csv")
        inter_obs_out_df = pd.DataFrame(obs_inter_comp_out, columns = ['task', 'intervene', 'trial_no', 'output', 'acc',])
        inter_obs_res_df = pd.DataFrame(obs_inter_comp_res, columns = ['task', 'intervene', 'trials', 'acc', 'std'])
        inter_obs_out_df.to_csv(inter_obs_comp_out_file)
        inter_obs_res_df.to_csv(inter_obs_comp_res_file)
                # get all trials for a particular causal relation from obs
                # map it to corresponding indices of inter 
                # have custom split by intervention or use reshape
                # store respones = 

def compare_obs_sub_perf(res_files, sample_size):
    obs_fold_template = "obs_%s"
    inter_fold_template = "sub_%s"
    
    for f in res_files:
        obs_inter_comp_res = []
        obs_inter_comp_out = []
        for task_id in range(TASKS):
            obs_fold = os.path.join(f, obs_fold_template % (task_id), "response_*.csv")
            inter_fold  = os.path.join(f, inter_fold_template % (task_id), "response_*.csv")
            obs_files = sorted(glob.glob(obs_fold))
            inter_files = sorted(glob.glob(inter_fold))
            obs_arrs = []
            inter_arrs = []
            for trial_no, (obs_file, inter_file) in enumerate(zip(obs_files, inter_files)):
                entry = [task_id, ]
                print(f"task_id: {task_id}, trial_no: {trial_no}")
                obs_resp = pd.read_csv(obs_file, sep = "|", index_col=0)
                inter_resp = pd.read_csv(inter_file, sep = "|", index_col=0)
                obs_arr = (obs_resp["label"] == obs_resp["pred"]).astype(int).to_numpy().reshape((sample_size, -1))
                inter_arr =  (inter_resp["label"] == inter_resp["pred"]).astype(int).to_numpy().reshape((sample_size, -1))
                nobs_arr = obs_arr[:, OBS_INTER_MAP[task_id]]
                obs_arrs.append(nobs_arr)
                inter_arrs.append(inter_arr)
            obs_arr = np.concatenate(obs_arrs, axis=0)
            inter_arr = np.concatenate(inter_arrs, axis=0)
            #print("obs_inter", obs_arr.shape, inter_arr.shape, obs_arr, inter_arr)
            res = (inter_arr & obs_arr).reshape(15, NUM_INTERS[task_id], -1).swapaxes(0,1)
            res_acc = res.mean(axis=-1)
            res_m, res_std = np.mean(res_acc, axis=-1), np.std(res_acc, axis=-1)
            #print(res.shape, res_acc.shape, res)
            trials = res.shape[1]
            for i, inter_var in enumerate(INTER_VARS[task_id]):
                for trial_no in range(trials):
                    obs_inter_comp_out.append([task_id, inter_var, trial_no, res[i][trial_no], res_acc[i][trial_no]])
                obs_inter_comp_res.append([task_id, inter_var, trials, res_m[i], res_std[i]])
        inter_obs_comp_out_file = os.path.join(f, "obs_sub_output.csv")
        inter_obs_comp_res_file = os.path.join(f, "obs_sub_results.csv")
        inter_obs_out_df = pd.DataFrame(obs_inter_comp_out, columns = ['task', 'intervene', 'trial_no', 'output', 'acc',])
        inter_obs_res_df = pd.DataFrame(obs_inter_comp_res, columns = ['task', 'intervene', 'trials', 'acc', 'std'])
        inter_obs_out_df.to_csv(inter_obs_comp_out_file)
        inter_obs_res_df.to_csv(inter_obs_comp_res_file)
                # get all trials for a particular causal relation from obs
                # map it to corresponding indices of inter 
                # have custom split by intervention or use reshape
                # store respones = 

def compute_pr(res_file, cm_map):
    df = pd.read_csv(res_file)
    df["output"] = df["output"].apply(lambda x: list(map(int, x.strip("[]").split(" "))))
    ndf = df.groupby(["task", "trial_no"])["output"].apply(lambda x: np.concatenate(x.to_list()).tolist())
    res = []
    for task_id in range(TASKS):
        out = np.array(ndf.loc[task_id].to_list())
        rel_c, rel_i = out[:, cm_map[task_id][0]],out[:, cm_map[task_id][1]]
        rel = out[:, cm_map[task_id][0] + cm_map[task_id][1]]
        avg = rel.mean() if rel.size > 0 else 0
        c_avg = rel_c.mean() if rel_c.size > 0 else 0
        i_avg = rel_i.mean() if rel_i.size > 0 else 0
        res.append((avg, c_avg, i_avg))
    return res