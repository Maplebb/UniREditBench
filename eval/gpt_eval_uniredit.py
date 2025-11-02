# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import argparse
import os
import os.path as osp
from typing import Callable, Iterable
import time
import re
from openai import AzureOpenAI
import pandas as pd
from utils import *



def resolve_after_image(output_dir: str, name: str, idx) -> str:
    path = osp.join(output_dir, str(name), f"{idx}.png")
    if not osp.exists(path):
        raise FileNotFoundError(f"Cannot find edited image at: {path}")
    return path

def _resolve_img(path_or_none: str) -> str:
    if not path_or_none:
        raise FileNotFoundError("Empty image path.")
    if not osp.exists(path_or_none):
        raise FileNotFoundError(f"Image not found: {path_or_none}")
    return path_or_none

def gpt_generate(inputs, temperature=0, max_tokens=4096, image_size=768, **kwargs):
    input_msgs = prepare_inputs(inputs, image_size=image_size)
    temperature = kwargs.pop('temperature', temperature)
    max_tokens = kwargs.pop('max_tokens', max_tokens)
    retries = 5
    for attempt in range(1, retries + 1):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                stream=False,
                messages=input_msgs,
                max_tokens=max_tokens,
                n=1,
                temperature=temperature,
            )
            response = response.to_dict()
            break
        except Exception as e:
            print(f'{inputs}')
            print(f"❌ [Attempt {attempt}/{retries}] Unexpected error: {e}")
            if attempt==retries:
                raise e
            time.sleep(3)

    ret_code = getattr(response, "status_code", 0) or 0
    ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code

    answer = 'Failed to obtain answer via API. '
    try:
        answer = response['choices'][0]['message']['content'].strip()
        print(f"Input: {input}\nAnswer: {answer}")
    except Exception as err:
        print(f'{type(err)}: {err}')
        print(response)
    return ret_code, answer, response

def track_progress_rich(
        func: Callable,
        tasks: Iterable = tuple(),
        nproc: int = 1,
        save=None,
        keys=None,
        batch_save: int = 10,   
        **kwargs) -> list:

    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    if save is not None:
        assert osp.exists(osp.dirname(save)) or osp.dirname(save) == ''
        if not osp.exists(save):
            dump({}, save)
    if keys is not None:
        assert len(keys) == len(tasks)
    if not callable(func):
        raise TypeError('func must be a callable object')
    if not isinstance(tasks, Iterable):
        raise TypeError(f'tasks must be an iterable object, but got {type(tasks)}')
    assert nproc > 0, 'nproc must be a positive number'

    res = load(save) if save is not None else {}
    results = [None for _ in range(len(tasks))]

    pending_since_last_dump = 0

    with ThreadPoolExecutor(max_workers=nproc) as executor:
        futures = []

        for inputs in tasks:
            if not isinstance(inputs, (tuple, list, dict)):
                inputs = (inputs,)
            if isinstance(inputs, dict):
                future = executor.submit(func, **inputs)
            else:
                future = executor.submit(func, *inputs)
            futures.append(future)

        unfinished = set(range(len(tasks)))
        pbar = tqdm(total=len(unfinished))

        try:
            while len(unfinished):
                new_finished = set()

                for idx in list(unfinished):
                    if futures[idx].done():
                        exception = futures[idx].exception()
                        if exception is not None:
                            results[idx] = {"error": repr(exception), "judge1": "", "judge2": "", "judge3": ""}
                            new_finished.add(idx)
                            if keys is not None:
                                res[keys[idx]] = results[idx]
                            continue  
                        else:
                            results[idx] = futures[idx].result()
                            new_finished.add(idx)
                            if keys is not None:
                                res[keys[idx]] = results[idx]

                if new_finished:
                    pbar.update(len(new_finished))
                    for k in new_finished:
                        unfinished.remove(k)

                    if save is not None:
                        pending_since_last_dump += len(new_finished)
                        if pending_since_last_dump >= batch_save:
                            dump(res, save)
                            pending_since_last_dump = 0

                time.sleep(0.1)
        finally:
            pbar.close()
            if save is not None and pending_since_last_dump > 0:
                dump(res, save)

    if save is not None:
        dump(res, save)

    return results


def find_image(output_dir, index):
    for suffix in ['png', 'jpg', 'jpeg']:
        img_path = osp.join(output_dir, f"{index}.{suffix}")
        if osp.exists(img_path):
            return img_path
    raise FileNotFoundError(f"Cannot find output images {index} in {output_dir}!!!")

def eval_unified(item, input_dir, output_dir, **kwargs):
    instruct = item['instruction']
    rules = item.get('rules', '')
    name = item['name']
    idx = item['idx']
    img1 = os.path.join(input_dir, item['original_image_path'])  
    ref_img = _resolve_img(os.path.join(input_dir, item['reference_image_path']))
    if isinstance(rules, str) and rules:
        instruct = rules + "\n" + instruct
        # print(rules)
    try:
        img2 = resolve_after_image(output_dir, name, idx) 
    except FileNotFoundError as e:
        return {"judge1": "", "judge2": "", "judge3": "", "error": str(e)}  

    prompt_cons = prompt_consist.format(instruct=instruct)
    reference = item.get('reference_effect', '') or ''
    prompt_rea = prompt_reasoning_w_ref_image.format(instruct=instruct, reference=reference)
    prompt_qua = prompt_generation

    msg1 = [
        {'type': 'text', 'value': prompt_cons},
        {'type': 'image', 'value': img1},
        {'type': 'image', 'value': img2},
    ]
    _, judge1, _ = gpt_generate(msg1, **kwargs)

    msg2 = [
        {'type': 'text', 'value': prompt_rea},
        {'type': 'image', 'value': img1},
        {'type': 'image', 'value': ref_img},
        {'type': 'image', 'value': img2},
    ]
    _, judge2, _ = gpt_generate(msg2, **kwargs)

    msg3 = [
        {'type': 'text', 'value': prompt_qua},
        {'type': 'image', 'value': img2},
    ]
    _, judge3, _ = gpt_generate(msg3, **kwargs)

    return dict(judge1=judge1, judge2=judge2, judge3=judge3)


def extract(answer):
    matches = re.findall(r'\*?\*?Final Score\*?\*?:?\s*([\d*\s,\n]*)', answer, re.IGNORECASE)
    numbers = []
    if matches:
        for match in matches:
            extracted_numbers = re.findall(r'\d+', match.replace('\n', ' '))
            if extracted_numbers:
                numbers.extend(map(int, extracted_numbers))
                break
        if numbers != []:
            return numbers

    matches = re.findall(r'\*?\*?Final Scores\*?\*?:?\s*([\d*\s,\n]*)', answer, re.IGNORECASE)
    numbers = []
    if matches:
        for match in matches:
            extracted_numbers = re.findall(r'\d+', match.replace('\n', ' '))
            if extracted_numbers:
                numbers.extend(map(int, extracted_numbers))
                break
        return numbers
    else:
        return None

def calculate_score(row):
    score = 0.3 * row['ApprConsistency'] + 0.5 * row['Reasoning'] + 0.2 * row['VisualPlausibility']
    return score

def calculate_completion(row):
    a, r, v = row['ApprConsistency'], row['Reasoning'], row['VisualPlausibility']
    if pd.isna(a) or pd.isna(r) or pd.isna(v):
        return float('nan')  
    return 1 if (a == 5 and r == 5 and v == 5) else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Json Path')
    parser.add_argument('--input', type=str, required=True, help='UniREditBench Image Dir')
    parser.add_argument('--output', type=str, required=True, help='Output Image Dir, outputs/MODEL_NAME')
    parser.add_argument('--prefix', type=str, default=None, help='output json prefix')
    parser.add_argument('--model', type=str, default="UniREdit-Bagel", help='Model Name')
    parser.add_argument('--nproc', type=int, default=8, help='n processes for api')

    args = parser.parse_args()

    model_name = args.output.split('/')[-1] if args.model is None else args.model
    if not args.prefix:
        tmp_file = f"{args.output}/{model_name}.pkl"
        judge_res = f"{args.output}/{model_name}_judge.xlsx"
        score_file = f"{args.output}/{model_name}_judge.csv"
    else:
        tmp_file = f"{args.output}/{args.prefix}_{model_name}.pkl"
        judge_res = f"{args.output}/{args.prefix}_{model_name}_judge.xlsx"
        score_file = f"{args.output}/{args.prefix}_{model_name}_judge.csv"

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    data = json.load(open(args.data))
    data = pd.DataFrame(data)

    data['key'] = data['name'].astype(str) + '#' + data['idx'].astype(str)

    result = {}
    if osp.exists(tmp_file):
        result = load(tmp_file)

    items = []
    for i in range(len(data)):
        row = data.iloc[i]
        if row['key'] not in result:
            items.append(row)

    tups = [dict(item=x, output_dir=args.output) for x in items]
    keys = [x['key'] for x in items]

    if len(tups):
        res = track_progress_rich(eval_unified, tups, nproc=args.nproc, chunksize=args.nproc, save=tmp_file, keys=keys)
        result = load(tmp_file)
        for k, v in zip(keys, res):
            if k not in result:
                result[k] = v

    judges = [result.get(k, {}) for k in data['key']]

    scores, judge_combine, judge_cons, judge_reas, judge_qua = [], [], [], [], []

    for judge in judges:
        s1 = extract(judge.get('judge1', '')) or []
        s2 = extract(judge.get('judge2', '')) or []
        s3 = extract(judge.get('judge3', '')) or []

        judge_cons.append(judge.get('judge1'))
        judge_reas.append(judge.get('judge2'))
        judge_qua.append(judge.get('judge3'))

        if len(s1) >= 1 and len(s2) >= 1 and len(s3) >= 1:
            scores.append([s1[0], s2[0], s3[0]])
        else:
            scores.append(None)

    reasoning, img_consist, gen_quality, match_log = [], [], [], []
    for score in scores:
        if score:
            match_log.append('succeed')
            img_consist.append(score[0])
            reasoning.append(score[1])
            gen_quality.append(score[2])
        else:
            match_log.append('failed')
            img_consist.append(None)
            reasoning.append(None)
            gen_quality.append(None)

    data['Reasoning'] = reasoning
    data['ApprConsistency'] = img_consist
    data['VisualPlausibility'] = gen_quality
    data['match_log'] = match_log
    data['judge_cons'] = judge_cons
    data['judge_reas'] = judge_reas
    data['judge_qua'] = judge_qua

    data['score'] = data.apply(calculate_score, axis=1)
    data['complete'] = data.apply(calculate_completion, axis=1)

    dump(data, judge_res)  

    # Overall
    score_final = data['score'].mean()
    completion_rate = data['complete'].mean()
    reasoning_average = data['Reasoning'].mean()
    img_consist_average = data['ApprConsistency'].mean()
    generation_quality = data['VisualPlausibility'].mean() 

    def trans_to_percent(s): return 25*(s-1)

    by_name = (
        data.groupby('name')
            .agg(score=('score', 'mean'),
                 complete=('complete', 'mean'))
            .reset_index()
    )
    by_name_dict = {
        str(r['name']): [r['score'], trans_to_percent(r['score']), r['complete']]
        for _, r in by_name.iterrows()
    }

    final_score = dict(
        Overall=[score_final, trans_to_percent(score_final), completion_rate],
        Overall_Reasoning=[reasoning_average, trans_to_percent(reasoning_average), None],
        Overall_ApprConsistency=[img_consist_average, trans_to_percent(img_consist_average), None],
        Overall_VisualPlausibility=[generation_quality, trans_to_percent(generation_quality), None],
        **{f"Class-{k}": v for k, v in by_name_dict.items()}
    )
    
    df = pd.DataFrame(final_score, index=["Score-Origin", "Score-Percentage", "Accuracy"]).T
    df.reset_index(inplace=True)
    df.columns = ["-", "Score-Origin", "Score-Percentage", "Accuracy"]
    df.to_csv(score_file, index=False)
    print(df)

    by_name_json = osp.join(args.output, f"{model_name}_by_name.json")
    with open(by_name_json, "w", encoding="utf-8") as f:
        json.dump(by_name_dict, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    base_url = "your_api_url"
    api_version = ""
    api_key = ""
    openai_client = AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=api_key,
    )
    main()
