import os
import argparse

from rbias.process import sep_responses_based_on_inter
from rbias.utils import get_taskwise_filters
from rbias.llm import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Generate Prompts')
    parser.add_argument("--model", '-m', type=str, required=True)
    parser.add_argument('--folders', "-f",  nargs='+', required=True)
    parser.add_argument("-run_id", '-i', type=str, required=True)
    args = parser.parse_args()
    model = args["model"]
    run_id = args["run_id"]
    files = [ os.path.join(folder_path, f"{model}_{run_id}") for folder_path in args["folders"] ]
    filters = get_taskwise_filters()
    sep_responses_based_on_inter(files, filters)
