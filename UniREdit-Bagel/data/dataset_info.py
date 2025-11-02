# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import ReasoningEditIterableDataset
# from .t2i_dataset import T2IIterableDataset
# from .vlm_dataset import SftJSONLIterableDataset
# from .interleave_datasets import UnifiedEditIterableDataset


DATASET_REGISTRY = {
    'reason_edit': ReasoningEditIterableDataset,
}

# 3principle_filtered_bg_pres
DATASET_INFO = {
    'reason_edit':{
        'reasonedit': {
            'data_dir': '/data_parquet',
            'num_files': 8, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 49937,
            "parquet_info_path": '/data_parquet/parquet_info.json', # number of total samples in the dataset
		},
    },
}

