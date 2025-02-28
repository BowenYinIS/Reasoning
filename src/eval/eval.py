# run under 4 V100-32GB GPUs

import os

import numpy as np

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # use mirror, only for chinese users
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # use 4 GPUs

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import importlib.util
import argparse
import random
import time
from datetime import datetime
from tqdm import tqdm
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.math_normalization import *
from utils.grader import *
import pickle
from math import comb
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_name_or_path', type=str, default="Qwen/QwQ-32B-Preview", help="model dir")
    parser.add_argument('--model_cache_dir', type=str, default="/root/autodl-tmp/QwQ", help="cache dir")
    # benchmark data
    parser.add_argument("--data_dir", default="/root/Reasoning/data/datasets", type=str)
    parser.add_argument('--data_name', type=str, default="aime", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument('--start_idx', type=int, default=0, help="data[start:end]")
    parser.add_argument('--end_idx', type=int, default=-1, help="data[start:end], if -1, data[start:]")
    # generation
    parser.add_argument('--n_sampling', type=int, default=1, help="n for sampling")
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens", default=3000, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    # prompt
    parser.add_argument("--prompt_type", default="qwen-instruct", type=str)
    parser.add_argument("--surround_with_messages", default=True, type=bool)
    parser.add_argument("--use_few_shot", default=False, type=bool)

    # output & seed
    parser.add_argument("--output_dir", default="/root/autodl-tmp/outputs", type=str)  # 存在数据盘
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dtype", default='auto', type=str)
    args = parser.parse_args()
    
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy
    return args


def get_conversation_prompt_by_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text


def get_three_prompt(prompt_type, data_name):
    file_path = os.path.join(".", "prompts", prompt_type, f"{data_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    # 动态导入模块
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'system_prompt'):
        system_prompt = module.system_prompt
    else:
        raise AttributeError(f"'system_prompt' not found in {file_path}")
    
    if hasattr(module, 'few_shot_prompt'):
        few_shot_prompt = module.few_shot_prompt
    else:
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")
    
    if hasattr(module, 'question_format'):
        question_format = module.question_format
    else:
        raise AttributeError(f"'question_format' not found in {file_path}")

    return system_prompt, few_shot_prompt, question_format


def infer(args):
    print(f"Loading model: {args.model_name_or_path}")

    # save file parameters
    # # model cache dir
    os.makedirs(args.model_cache_dir, exist_ok=True)
    # # output dir
    model_name = "/".join(args.model_name_or_path.split("/")[-3:])
    output_file_path = (f'{args.output_dir}/{model_name}/{args.data_name}/'
                        f'{args.split}_{args.prompt_type}_t{args.temperature}_n{args.n_sampling}_s{args.start_idx}_e{args.end_idx}.jsonl')
    os.makedirs(f'{args.output_dir}/{model_name}/{args.data_name}', exist_ok=True)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 cache_dir=args.model_cache_dir,
                                                 device_map='balanced_low_0',
                                                 torch_dtype='auto',
                                                 output_attentions=True)

    # load benchmark data
    print(f"Loading data: {args.data_name} {args.split}")
    examples = load_data(args.data_name, args.split, args.data_dir)
    if args.end_idx == -1:
        args.end_idx = len(examples)
    examples = examples[args.start_idx:args.end_idx]

    # Main Loop
    print(f"Start inference: {args.data_name} {args.split}")
    system_prompt, few_shot_prompt, question_format = get_three_prompt(args.prompt_type, args.data_name)
    for example in tqdm(examples, total=len(examples)):
        # skip if the example has been processed
        example_id = example["id"]
        output_attention_path = output_file_path.replace(".jsonl", f"{example_id}_attentions.pt")
        # output_logits_path = output_file_path.replace(".jsonl", f"{example_id}_logits.pt")
        if os.path.exists(output_file_path) and os.path.exists(output_attention_path):
            continue

        # parse question and answer
        question = parse_question(example, args.data_name)
        gt_cot, gt_ans = parse_ground_truth(example, args.data_name)

        # construct prompt
        if args.use_few_shot:
            cur_prompt = few_shot_prompt + question_format.format(question=question)
        else:
            cur_prompt = question_format.format(question=question)
        if args.surround_with_messages:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cur_prompt}
            ]
            cur_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)

        with torch.no_grad():
            inputs = tokenizer([cur_prompt], return_tensors="pt").to('cuda:0')  # 数据加载到第一块GPU
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                top_p=args.top_p,
                do_sample=False,
                num_return_sequences=args.n_sampling,  # TODO: 检查多个sample下以下代码运行是否正确；实现temperature=0时的处理
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_attentions=True,
                # output_logits=True,  # 节约存储空间，不输出
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs.sequences)
            ]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            generated_answer = extract_answer(responses, args.data_name)
            is_correct = check_is_correct(generated_answer, gt_ans)

        # save to file
        print(f"Save example ID: {example_id}")
        example["generated_response"] = responses
        example["generated_answer"] = generated_answer
        example["is_correct"] = is_correct
        with open(output_file_path, "a", encoding="utf-8") as f:
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")

        saved_attention = [step[-1].float().cpu() for step in outputs.attentions]
        saved_generated_ids = [i.cpu() for i in generated_ids]
        saved_generated_tokens = [tokenizer.convert_ids_to_tokens(i) for i in saved_generated_ids]


        torch.save({
            'input_ids': inputs.input_ids.cpu(),
            "generated_ids": saved_generated_ids,
            'input_tokens': tokenizer.convert_ids_to_tokens(inputs.input_ids[0]),  # 仅用于debug
            "generated_tokens": saved_generated_tokens,
            "attentions": saved_attention,
        }, output_attention_path)
        # torch.save([[att.float().cpu() for att in atts] for atts in outputs.attentions], output_attention_path)
        # torch.save([logits.float().cpu() for logits in outputs.logits], output_logits_path)

        # 回收显存
        del inputs, outputs, generated_ids, responses, generated_answer, is_correct
        torch.cuda.empty_cache()

        # TODO： 计算pass@k


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    infer(args)
