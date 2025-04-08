# Effective and Efficient Masked Image Generation Models<br><sub>Official PyTorch Implementation</sub>


## Preparation

### Dataset
Download [ImageNet](http://image-net.org/download) dataset, and place it in your `IMAGENET_PATH`.

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/ML-GSAI/eMIGM.git
cd eMIGM
```

2. **Set up environment**:
```bash
bash init_env.sh
```

3. **Download pretrained models**:
   - **VAE** (for 256×256 generation):  
     [Download link](https://www.dropbox.com/scl/fi/hhmuvaiacrarfg28qxhwz/kl16.ckpt?rlkey=l44xipsezc8atcffdp4q7mwmh&dl=0)
   - **DC-AE** (for 512×512 generation):  
     [Hugging Face model card](https://huggingface.co/mit-han-lab/dc-ae-f32c32-in-1.0)

4. **Prepare FID reference statistics**:
   - Download [`fid_stats`](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP)
   - Place the directory in the project root as `fid_stats/`  
   
   *Note: For 256×256 generation, you only need the `fid_stats_imagenet256_guided_diffusion.npz` file. The `fid_stats_imagenet512_guided_diffusion.npz` file is required only for 512×512 generation.*
   
   For the `fid_stats_imagenet256_guided_diffusion.npz` file, you need to convert the data type of `mu` to float32 using the following script:
   ```python
    import numpy as np
    # Load the npz file
    data = np.load('fid-stats/fid_stats_imagenet256_guided_diffusion.npz')
    # Extract data, convert to float32, and save
    mu = data['mu'].astype(np.float32)
    sigma = data['sigma']
    np.savez('fid-stats/fid_stats_imagenet256_guided_diffusion.npz', mu=mu, sigma=sigma)
   ```

<small>We gratefully acknowledge the original authors for providing the pretrained models.</small>

## Accelerate Training with Latent Caching
To optimize training efficiency, we recommend pre-caching the latent representations of your ImageNet dataset. This preprocessing step significantly reduces computational overhead during the training process:

**For 256x256 resolution** (using VAE):
```bash
# Generate latent representations using pre-trained VAE
bash bash/extract/extract_imagenet_256.sh
```

**For 512x512 resolution** (using DC-AE):
```bash
# Generate latent representations using pre-trained DC-AE 
bash bash/extract/extract_imagenet_512.sh
```

These scripts will pre-process the dataset into compressed latent representations that can be loaded efficiently during model training. We strongly recommend running this caching process before initiating the main training procedure. In our standard workflow, we always perform this preprocessing step prior to training as a best practice.

## Usage
**Supported Model Sizes**:
- **256×256 Resolution**: Five model variants available  
  `eMIGM-MS`, `eMIGM-S`, `eMIGM-B`, `eMIGM-L`, `eMIGM-H`
- **512×512 Resolution**: Four model variants available  
  `eMIGM-MS`, `eMIGM-S`, `eMIGM-B`, `eMIGM-L`

### Training Instructions

We provide dedicated training scripts for both 256×256 and 512×512 resolutions:

**Configuration and Execution**:

1.  **Script Locations**:
    *   Training scripts for 256x256 models are located in: `bash/train/256`
    *   Training scripts for 512x512 models are located in: `bash/train/512`

2.  **Example Command (256x256, eMIGM-S)**:

    ```bash
    # Train the eMIGM-S model at 256x256 resolution.
    bash bash/train/256/train_emigm_small.sh
    ```

3.  **Parameter Configuration**: Before running the training scripts, you **must** modify the following parameters within the script:

    *   `output_dir`:  The directory where training outputs (checkpoints, logs) will be saved.
    *   `nnodes`:      The number of distributed training nodes.
    *   `nproc_per_node`: The number of processes to launch per node (typically, the number of GPUs per node).
    *   `data_path`:    The path to your ImageNet dataset.
    *   `cached_path`:  The path to the pre-cached latent representations (created by the extraction scripts, e.g., `extract_imagenet_256.sh`).  This is crucial for efficient training.
    *   `vae_path`:     The path to the pretrained VAE or DC-AE model.

> **Note**: The `cached_path` should match the output directory specified in your `extract_imagenet_256.sh` or `extract_imagenet_512.sh` execution. The `vae_path` should be the path to the pretrained VAE or DC-AE model, for 256x256 resolution, you can use the pretrained VAE model, and for 512x512 resolution, you can use the pretrained DC-AE model.

### Evaluation Instructions

Pre-trained models are available for download at: [https://huggingface.co/GSAI-ML/eMIGM](https://huggingface.co/GSAI-ML/eMIGM)

We provide dedicated evaluation scripts for both 256×256 and 512×512 resolutions:

**Configuration and Execution**:

1.  **Script Locations**:
    *   Evaluation scripts for 256x256 models: `bash/evaluate/evaluate_256.sh`
    *   Evaluation scripts for 512x512 models: `bash/evaluate/evaluate_512.sh`

2.  **Example Command**:
    ```bash
    # Evaluate a model at 256x256 resolution
    bash bash/evaluate/evaluate_256.sh
    ```

3.  **Parameter Configuration**: Before running the evaluation scripts, you **must** modify the following parameters:
    *   `ckpt_path`:    The path to the downloaded pre-trained model checkpoint.
    *   `output_dir`:   The directory where evaluation outputs and logs will be saved.
    *   `nnodes`:       The number of distributed evaluation nodes.
    *   `nproc_per_node`: The number of processes per node (typically the number of GPUs per node).
    *   `data_path`:    The path to your ImageNet dataset.
    *   `vae_path`:     The path to the pretrained VAE (for 256x256) or DC-AE model (for 512x512).

> **Note**: For 256x256 resolution evaluations, use the pretrained VAE model. For 512x512 resolution evaluations, use the pretrained DC-AE model.

## Acknowledgements
> A large portion of codes in this repo is based on [MAR](https://github.com/LTH14/mar) and [DPM-Solver](https://github.com/LuChengTHU/dpm-solver).

## Contact

> If you have any questions, feel free to contact me through email (zebin@ruc.edu.cn). Enjoy using eMIGM!