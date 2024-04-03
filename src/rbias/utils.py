import numpy as np
import pandas as pd

def load_template(template_path):
    sep = '|' # based on the var_type
    return pd.read_csv(template_path, sep = sep, index_col=False)

def load_data(data_path):
    sep = ','
    return pd.read_csv(data_path, sep = sep, index_col = False)

def get_taskwise_filters():
    t_filters = [[[1], [0]], [[0, 1, 2, 4, 5,7, 8], [3, 6]], [[0, 1, 2, 4, 6], [3, 5, 7, 8]]]
    n_queries = [2, 9, 9]
    filters = []
    for i, task in enumerate(t_filters):
        g_inter, b_inter = task[0], task[1]
        g_inter = np.array(g_inter)
        b_inter = np.array(b_inter)
        print(g_inter, b_inter)
        q = n_queries[i]
        g_res = np.concatenate([g_inter, g_inter + q, g_inter + 2*q])
        b_res = np.concatenate([b_inter, b_inter + q, b_inter + 2*q])
        print(g_res, b_res)
        filters.append((g_res, b_res))
    return filters