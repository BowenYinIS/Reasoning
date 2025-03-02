import argparse

def parse_list(arg):
    return arg.split(',')

def parse_args_local():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--n_sampling', type=int, default=1, help="n for sampling")
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument('--start_idx', type=int, default=0, help="data[start:end]")
    parser.add_argument('--end_idx', type=int, default=-1, help="data[start:end], if -1, data[start:]")
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens", default=2048, type=int)
    parser.add_argument("--prompt_type", default="qwen-base", type=str)
    parser.add_argument("--prompt_file_path", default="./prompts", type=str)
    parser.add_argument("--surround_with_messages", action="store_true")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument('--stop', type=parse_list)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dtype", default='auto', type=str)
    parser.add_argument("--completions_save_dir", default='./completions', type=str)
    # parser.add_argument("--use_qwen_check", action="store_true")
    args = parser.parse_args()
    
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy 
    print(f"current stop list: {args.stop}")
    return args