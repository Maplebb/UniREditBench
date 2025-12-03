# first modify data/dataset_info.py
num_nodes=1
node_rank=0
master_addr=127.0.0.1
master_port=29502
model_path=/bagel-7b-mot
llm_path=/Qwen2.5-0.5B-Instruct
vit_path=/siglip-so400m-14-980-flash-attn2-navit
output_checkpoint_dirpath=/checkpoint
results_dir=/log_dir

torchrun \
  --nnodes=$num_nodes \
  --nproc_per_node=8 \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --num_shard=8 \
  --text_cond_dropout_prob=0 \
  --vit_cond_dropout_prob=0 \
  --model_path "$model_path" \
  --llm_path "$llm_path" \
  --vit_path "$vit_path" \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from "$model_path" \
  --results_dir "$results_dir" \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --ema 0.995 \
  --warmup_steps 1000 \
  --total_steps 15000 \
  # We finally adopted the checkpoint after 5,000 steps of fine-tuning.
  --min_lr 1e-6 \
  --lr_scheduler cosine \
  --save_every 1000 \
  --ce_weight 2 \
  --lr 2e-5 \
  --timestep_shift 4 \
  --num_worker 1 \
  --expected_num_tokens 35000 \
  --max_num_tokens 35000 \
  --max_num_tokens_per_sample 35000 \
  --checkpoint_dir "$checkpoint_dir" \
  # --cpu_offload True
