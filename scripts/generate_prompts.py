import os
import time
import argparse


from rbias import generate_prompts, generate_prompts_v2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Generate Prompts')
    parser.add_argument("--var_type", '--vt', choices=['rchar', 'cs', 'adv'], 
                        required=True)
    parser.add_argument("--trials", "-t", type =int)
    parser.add_argument("--var_cnt", "--vc", type=int)
    parser.add_argument("--dest_path", '--dp', type =str)
    args=vars(parser.parse_args())
    print("Starting to generate prompts...")
    print(args['var_type'], args['trials'], args['var_cnt'], args['dest_path'])
    #dest_path = os.path.join(os.path.abspath('.'), args['dest_path'])
    var_map = [
                    (['<T1>', '<T2>', '<vT1>', '<vT2>']),
                    (['<T1>', '<T2>', '<T3>', '<vT1>', '<vT2>', '<vT3>']),
                    (['<T1>', '<T2>', '<T3>', '<vT1>', '<vT2>', '<vT3>']),
                    (['<T1>', '<T2>', '<T3>', '<vT1>', '<vT2>', '<vT3>'])
                ]
    generate_prompts_v2(args['var_type'], args['trials'], args['var_cnt'], 
                        var_map, args['dest_path'])
    # we need src data path, src prompt template
    # variable type -  rchar, cs or acs
    # based on the variable type infer the path of the data and the prompt.
    # read data file, load prompt tempalate
    # generate prompts with right context, map to the ground truth
    # include number of trials, no of prompts per trial - number of prompts x number of variables. 
    # store the prompts in folder - trials/task/prompts.csv
