#!/bin/bash
# Set NCCL environment variables to control log level
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

exp_name=imagenet512_emigm_large_d8w1280
train_word="--model emigm_large --diffloss_d 8 --diffloss_w 1280  --save_epoch_name True --mask_strategy exp --use_mae True --clamp_t_min 0.2 --use_unsup_cfg True --batch_size 64 --evaluate_multi --num_images 10000 --cfg 2.9 --cfg_schedule linear --temperature 0.95 --eval_bsz 256"
# we using 64 batch size per gpu and uisng 32 gpus, so the total batch size is 64 * 32 = 2048

# need to be set
output={output_dir} # the path of output directory
nnodes={nnodes} # the number of nodes
nproc_per_node={nproc_per_node} # the number of processes per node
data_path={data_path} # the path of imagenet dataset, if using cached dataset, this path will be ignored
cached_path={cached_path} # the path of cached dataset, using this will accelerate the training, if using cached dataset, --use_cached must be added
vae_path={vae_path} # the path of dc ae checkpoint such as checkpoint/dc-ae-f32c32-in-1.0

mkdir -p "$output"/"$exp_name"
echo "$train_word" > "${output}/${exp_name}/train_params.txt"

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

echo "master_addr ${MASTER_ADDR}"
echo "master_port ${MASTER_PORT}"
echo "node_rank ${RANK}"

torchrun --nnodes $nnodes --nproc_per_node $nproc_per_node --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $RANK main_emigm.py \
--img_size 512 --vae_path $vae_path --vae_embed_dim 32 --vae_stride 32 --patch_size 1 \
--epochs 800 --warmup_epochs 100 --blr 1e-4 --diffusion_batch_mul 4 \
--online_eval \
--output_dir "$output"/"$exp_name" --resume "$output"/"$exp_name" \
--data_path $data_path \
--use_cached --cached_path $cached_path \
--num_workers 32 --eval_freq 50 $train_word

