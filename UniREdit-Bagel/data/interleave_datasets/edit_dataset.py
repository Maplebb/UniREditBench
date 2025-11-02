# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import random
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb
import random


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        image_num = len(row["image_list"])
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = random.choice(range(image_num - 1))
        max_end = min(start_idx + 3, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))

        data = self._init_data()
        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][start_idx]))),
            need_loss=False, 
            need_vae=True, 
            need_vit=True, 
        )

        if end_idx - start_idx > 1 and random.random() < 0.5: # concat multiple insturction
            if end_idx == image_num - 1:
                end_idx -= 1

            instruction = ""
            for idx in range(start_idx + 1, end_idx + 1):
                instruction += random.choice(row["instruction_list"][idx-1]) + ". "
            data = self._add_text(data, instruction.rstrip(), need_loss=False)
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][end_idx]))),
                need_loss=True, 
                need_vae=False, 
                need_vit=False,
            )
        else:
            for idx in range(start_idx + 1, end_idx + 1):
                instruction = random.choice(row["instruction_list"][idx-1])
                data = self._add_text(data, instruction, need_loss=False)
                if idx != end_idx:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True, 
                        need_vae=True, 
                        need_vit=True,
                    )
                else:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True, 
                        need_vae=False, 
                        need_vit=False,
                    )
        return data


class ReasoningEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        prompt = "You should first think about the planning process in the mind and then generate the image. The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here"
        before_image_path = row["before_image_path"]
        after_image_path = row["after_image_path"]
        instruction = row["instruction"]
        reasoning = row["reasoning"]
        reasoning = f"<think>{reasoning.strip()}</think>"
        
        # print(after_image_path,'\n',reasoning)
        # randomly choose start and end, return [0, 1] when only two images
        x = random.random()
        if x<0.2:
            if "unchanged_instruction" in row.keys() and "unchanged_reasoning" in row.keys():
                instruction = row["unchanged_instruction"]
                reasoning = row["unchanged_reasoning"]
                reasoning = f"<think>{reasoning.strip()}</think>"
                after_image_path = before_image_path
        print(after_image_path,'\n',reasoning)
            

        data = self._init_data()
        data = self._add_text(data, prompt, need_loss=False)
        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(before_image_path)),
            need_loss=False, 
            need_vae=True, 
            need_vit=True, 
        )

        data = self._add_text(data, instruction.rstrip(), need_loss=False)
        
        data = self._add_text(data, reasoning.rstrip(), need_loss=True)
        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(after_image_path)),
            need_loss=True, 
            need_vae=False, 
            need_vit=False,
        )
        return data