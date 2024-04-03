import os
import numpy as np
import pandas as pd
import argparse


def plot_results(model_name, result_paths):
    res = []
    for result_path in result_paths:
        res_path = os.path.join(result_path, f"{model_name}_1", "results.csv")
        res_df = pd.read_csv(res_path)
        accs = res_df['acc'].to_numpy()
        print(accs.shape)
        res.append(accs[:30])
    
    res_arr = np.array(res)
    res_arr = np.reshape(res, (3, 6, 5))
    print(res_arr, res_arr.shape)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Plot Results')
    parser.add_argument("--model", '-m', type=str, required=True)
    parser.add_argument("--results", '-r', nargs='+', required=True)
    args = vars(parser.parse_args())
    print(args['model'], args['results'])
    plot_results(args['model'], args['results'])