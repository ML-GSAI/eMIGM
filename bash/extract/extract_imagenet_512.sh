vae_path={vae_path} # the path of dc ae checkpoint such as checkpoint/dc-ae-f32c32-in-1.0
data_path={data_path} # the path of imagenet dataset
cached_path={cached_path} # the path of cached dataset, you can use the cached dataset to accelerate the evaluation

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    main_cache_dc.py \
    --img_size 512 --vae_path $vae_path \
    --batch_size 128 \
    --data_path $data_path --cached_path $cached_path

