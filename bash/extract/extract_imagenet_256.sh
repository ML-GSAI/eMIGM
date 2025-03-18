vae_path={vae_path} # the path of vae checkpoint such as checkpoints/kl16.ckpt
data_path={data_path} # the path of imagenet dataset
cached_path={cached_path} # the path of cached dataset, you can use the cached dataset to accelerate the evaluation

torchrun --nnodes 1 --nproc_per_node 8 --node_rank=0 \
    main_cache.py \
    --img_size 256 --vae_path $vae_path --vae_embed_dim 16 \
    --batch_size 128 \
    --data_path $data_path --cached_path $cached_path
