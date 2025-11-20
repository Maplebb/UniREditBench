set -x

GPUS=8
model_path=/UniREdit-Bagel_checkpoint
input_path=./UniREditBench
output_path=/output_dir

# generate images
torchrun \
    --nnodes=1 \
    --nproc_per_node=$GPUS \
    gen_images_mp_uniredit.py \
    --input_dir $input_path \
    --output_dir $output_path \
    --metadata_file /data_json \
    --max_latent_size 64 \
    --model-path $model_path \
    --think

sleep 60

