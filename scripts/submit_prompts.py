import os
import time
import argparse
import warnings
# warnings.filterwarnings("ignore")

from rbias import run_prompts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Query LLM')
    parser.add_argument("--model_name", "-m", type=str, 
                        required=True)
    parser.add_argument("--prompt_path", '-p', type=str, 
                        required=True)
    parser.add_argument("--result_path", "-r", type=str, 
                        required=True)
    parser.add_argument("--run_id", "-i", type=int, 
                        required=True)
    args=vars(parser.parse_args())
    print("Querying the prompts...")
    print(args['model_name'], args['prompt_path'], args['result_path'], args['run_id'])
    #dest_path = os.path.join(os.path.abspath('.'), args['dest_path'])

    run_prompts(args['model_name'], args['prompt_path'], args['result_path'], args['run_id'])
