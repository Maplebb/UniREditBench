# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import base64
import io
import json
import pandas as pd
import pickle
import csv


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)

def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth, **kwargs):
        json.dump(data, open(pth, 'w'), indent=4, ensure_ascii=False, cls=NumpyEncoder)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False, cls=NumpyEncoder) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine='xlsxwriter')

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep='\t', index=False, encoding='utf-8', quoting=quoting)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f, **kwargs)


def load(f, fmt=None):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    if fmt is not None:
        return handlers[fmt](f)

    suffix = f.split('.')[-1]
    return handlers[suffix](f)


def encode_image_to_base64(img, target_size=-1, fmt='JPEG'):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode('utf-8')
    return ret


def prepare_itlist(inputs, image_size=-1,  **kwargs):
    assert np.all([isinstance(x, dict) for x in inputs])
    has_images = np.sum([x['type'] == 'image' for x in inputs])
    if has_images:
        content_list = []
        for msg in inputs:
            if msg['type'] == 'text':
                content_list.append(dict(type='text', text=msg['value']))
            elif msg['type'] == 'image':
                from PIL import Image
                img = Image.open(msg['value'])
                b64 = encode_image_to_base64(img, target_size=image_size)
                img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail='high')
                content_list.append(dict(type='image_url', image_url=img_struct))
    else:
        assert all([x['type'] == 'text' for x in inputs])
        text = '\n'.join([x['value'] for x in inputs])
        content_list = [dict(type='text', text=text)]
    return content_list


def prepare_inputs(inputs, system_prompt=None, **kwargs):
    input_msgs = []
    if system_prompt is not None:
        input_msgs.append(dict(role='system', content=system_prompt))
    assert isinstance(inputs, list) and isinstance(inputs[0], dict)
    assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
    if 'role' in inputs[0]:
        assert inputs[-1]['role'] == 'user', inputs[-1]
        for item in inputs:
            input_msgs.append(dict(role=item['role'], content=prepare_itlist(item['content'], **kwargs)))
    else:
        input_msgs.append(dict(role='user', content=prepare_itlist(inputs, **kwargs)))
    return input_msgs


prompt_consist = """You are a highly skilled image evaluator. You will receive two images (an original image and a modified image) along with a specific modification instruction. The second image is known to have been altered based on this instruction, starting from the first image. Your task is to evaluate whether the two images maintain consistency in aspects not related to the given instruction.

## Task
Evaluate the consistency between the images according to the following scale (1 to 5):

- **5 (Perfect Consistency)**: Apart from changes explicitly required by the instruction, all other details (e.g., personal features, clothing, background, layout, colors, positions of objects) are completely identical between the two images.

- **4 (Minor Differences)**: Apart from changes explicitly required by the instruction, the second image is mostly consistent with the original image but contains a minor discrepancy (such as a missing minor personal feature, accessory, or tattoo).

- **3 (Noticeable Differences)**: Apart from changes explicitly required by the instruction, the second image has one significant difference from the original (such as a noticeable alteration in a person's appearance like hair or skin color, or a significant change in background environment).

- **2 (Significant Differences)**: Apart from changes explicitly required by the instruction, the second image has two or more significant differences or multiple noticeable inconsistencies (such as simultaneous changes in both personal appearance and background environment).

- **1 (Severe Differences)**: Apart from changes explicitly required by the instruction, nearly all key details (e.g., gender, major appearance features, background environment, or scene layout) significantly differ from the original image, clearly deviating from the original.

Example:

Original image: A blue-and-white floral vase on a wooden stand near a sunlit window, with beige wall and curtains in the background.
Instruction: "Throw the baseball towards the vase with sufficient force to make a crack."

- **Score 5**: All non-instruction details are identical: same window panes and view, curtain shape, wooden stand design and wood grain, same lighting.
- **Score 4**: The window changes (e.g., pane geometry, framing, or outside view), while curtains, wooden stand, lighting.
- **Score 3**: Both the window and the curtains show noticeable changes; wooden stand, lighting.
- **Score 2**: The background as a whole is clearly different (e.g., wall tone/texture and window/curtain styling or lighting all shift), though the wooden stand still matches.
- **Score 1**: The background differs and the wooden stand also changes (design/color/material) or is missing; overall scene layout is largely different.

Note: When assigning scores, only consider details unrelated to the instruction. Changes explicitly requested by the instruction should NOT be regarded as inconsistencies.

## Input

**Instruction:** {instruct}

## Output Format

Provide a detailed, step-by-step explanation of your scoring process. Conclude clearly with the final score, formatted as:

**Final Score:** **1-5**"""


prompt_reasoning_w_ref_image = """You are an expert image evaluator. For each task, you will be provided with:


1. An **original image**. The image before image editing.
2. An **instruction** describing how the original image should be modified.
3. A **ground-truth textual description** that represents the intended result of the modification.
4. A **reference image**. This shows the intended visual effects of the result. It is a guide for how the result should look.
5. An **output image**. generated by an assistant.


Your task is to assess the output image based on the following evaluation dimension:

## Evaluation Dimension: Alignment Between Image and Reference Description
Assess how accurately the output image aligns with the literal text of the **ground-truth description**. You should also compare the output image with the reference image to help judge whether the intended visual effects have been successfully achieved.

**Scoring Criteria:**
- **5**: The image completely matches the description, accurately reflecting every detail and degree.
- **4**: The image mostly matches the description, with minor discrepancies.
- **3**: The image partially matches the description but contains differences or lacks some details.
- **2**: The image contains noticeable difference. Important details are missed or clearly inaccurate.
- **1**: The image fails to follow the instruction and does not correspond to the description at all.

**Example**
Instruction: Turn this image into a rainy evening.
Description: The street ground has become wet and shiny, and there are visible puddles reflecting lights from the streetlamps and shop windows. The sky is a dark, dim blue, and the streetlamps are turned on.
Reference Image: A photo of a different city street at night, capturing the intended rainy effect. It clearly shows a dark, moody blue sky, bright orange streetlights, and strong, sharp reflections of these lights on the wet ground.
- **5**: Wet/shiny ground, visible puddles with clear reflections, dark blue sky, and streetlamps are on.
- **4**: Wet/shiny ground and streetlamps are on, but no visible puddles or unclear reflections.
- **3**: Dark sky and streetlamps are on, but the ground is completely dry.
- **2**: Only a slightly darkened sky; ground remains dry and streetlamps are off.
- **1**: Image is still in daytime or incorrectly edited (e.g., shows snow).

## Input
****
**original image:**  The first image uploaded
**Instruction**  {instruct}
**GroundTruth Description:** {reference}
**reference image:** The second image uploaded.
**Output Image:** The third image uploaded.


## Output Format

Provide a detailed, step-by-step explanation of your scoring process. Conclude clearly with the final score, formatted as:

**Final Score:** **X**
"""


prompt_generation = """You are an expert image evaluator. For each task, you will be provided with an **output image** generated by an assistant.

Your task is to independently assess the image along the following dimension and assign an integer score from **1 to 5**:

### Evaluation Dimension: Realism and Generation Quality

Assess the overall visual realism and generation fidelity of the image. Consider the image’s clarity, natural appearance, and compliance with physical plausibility and real-world constraints.

**Scoring Guidelines:**

- **5** The image is sharp, visually coherent, and all elements appear highly realistic and physically plausible.
- **4** The image is clear, with most elements appearing realistic; minor details may show slight unreality.
- **3** The image is mostly clear, but some significant elements appear unrealistic or physically implausible.
- **2** The image is noticeably blurry or contains major unrealistic components or visual distortions.
- **1** The image is extremely blurry, incoherent, or severely unrealistic; realism is nearly absent.

## Output Format

After the evaluation, conclude clearly with the final score, formatted as:

**Final Score:** **X**
"""

