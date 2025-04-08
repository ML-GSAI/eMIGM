#!/bin/bash
# Set NCCL environment variables to control log level
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

# need to be set
ckpt={ckpt_path} # the path of checkpoint, for example eMIGM_small_256_checkpoint.pth
output={output_dir} # the path of output directory
nnodes={nnodes} # the number of nodes
nproc_per_node={nproc_per_node} # the number of processes per node
data_path={data_path} # the path of imagenet dataset, if using cached dataset, this path will be ignored
vae_path={vae_path} # the path of vae checkpoint such as checkpoints/kl16.ckpt

exp_name=$(basename "$ckpt" | sed 's/_checkpoint\.pth$//')
mkdir -p "$output"/"$exp_name"

# Extract model parameters from exp_name
model_type=$(echo "$exp_name" | cut -d'_' -f1)
model_size=$(echo "$exp_name" | cut -d'_' -f2)

# Set model parameters based on exp_name
if [ "$model_type" = "eMIGM" ]; then
    if [ "$model_size" = "xsmall" ]; then
        model_params="--model emigm_xsmall --diffloss_d 3 --diffloss_w 1024"
    elif [ "$model_size" = "small" ]; then
        model_params="--model emigm_small --diffloss_d 3 --diffloss_w 1024"
    elif [ "$model_size" = "base" ]; then
        model_params="--model emigm_base --diffloss_d 6 --diffloss_w 1024"
    elif [ "$model_size" = "large" ]; then
        model_params="--model emigm_large --diffloss_d 8 --diffloss_w 1280"
    elif [ "$model_size" = "huge" ]; then
        model_params="--model emigm_huge --diffloss_d 12 --diffloss_w 1536"
    else
        echo "Unknown model size: $model_size"
        exit 1
    fi
else
    echo "Unknown model type: $model_type"
    exit 1
fi

# Define evaluate_word list
# Common parameters for all model sizes
num_iter_1="16"
num_sampling_steps_1="8"
num_iter_2="128"
num_sampling_steps_2="14"

# Set cfg_t_min/max based on model size
if [ "$model_size" = "xsmall" ] || [ "$model_size" = "small" ] || [ "$model_size" = "large" ]; then
    cfg_t_min_1="0.1"
    cfg_t_max_1="0.3"
    cfg_t_min_2="0.18"
    cfg_t_max_2="0.6"
elif [ "$model_size" = "base" ]; then
    cfg_t_min_1="0.1"
    cfg_t_max_1="0.3"
    cfg_t_min_2="0.05"
    cfg_t_max_2="0.4"
elif [ "$model_size" = "huge" ]; then
    cfg_t_min_1="0.2"
    cfg_t_max_1="0.4"
    cfg_t_min_2="0.3"
    cfg_t_max_2="0.7"
fi

evaluate_words=(
"--eval_bsz 64 --num_images 50000 --use_unsup_cfg True --cfg_t_min $cfg_t_min_1 --cfg_t_max $cfg_t_max_1 --num_iter $num_iter_1 --num_sampling_steps $num_sampling_steps_1 --sampling_algo dpm-solver --cfg 4 --cfg_schedule constant --temperature 1.0 --mask_strategy exp"  
"--eval_bsz 64 --num_images 50000 --use_unsup_cfg True --cfg_t_min $cfg_t_min_2 --cfg_t_max $cfg_t_max_2 --num_iter $num_iter_2 --num_sampling_steps $num_sampling_steps_2 --sampling_algo dpm-solver --cfg 4 --cfg_schedule constant --temperature 1.0 --mask_strategy exp"  
# Add more evaluate_word combinations here
)

# Set necessary parameters for distributed training
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

echo "master_addr ${MASTER_ADDR}"
echo "master_port ${MASTER_PORT}"
echo "node_rank ${RANK}"

# Iterate through each evaluate_word and execute corresponding operations
for evaluate_word in "${evaluate_words[@]}"; do
  # Replace illegal characters with spaces (replace spaces with underscores) for folder naming
  folder_name=$(echo $evaluate_word | sed 's/--//g' | sed 's/ /_/g' | sed 's/=/-/g')
  folder_name="evaluate_${folder_name}"  # Add prefix
  folder_name="$output/$exp_name/$folder_name"

  # Run Python script
  torchrun --nnodes $nnodes --nproc_per_node $nproc_per_node --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $RANK main_emigm.py \
    --img_size 256 --vae_path $vae_path --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
    --epochs 800 --warmup_epochs 100 --batch_size 64 --blr 1.0e-4 --diffusion_batch_mul 4 \
    --output_dir "$output"/"$exp_name" --resume "$ckpt" \
    $model_params \
    --data_path $data_path \
    --evaluate \
    $evaluate_word > $folder_name.txt
done
