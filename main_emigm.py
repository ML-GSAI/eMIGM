import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc
import util.dist as dist
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.loader import CachedFolder

from models.vae import AutoencoderKL
from models import emigm
from engine_emigm import train_one_epoch, evaluate
import copy

def str2bool(value):
    value = value.lower()
    if value in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args_parser():
    parser = argparse.ArgumentParser('eMIGM training with Diffusion Loss', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=400, type=int)

    # Model parameters
    parser.add_argument('--model', default='emigm_large', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--mask_strategy', default='cosine', type=str)
    parser.add_argument('--use_weighted_loss', action='store_true', dest='use_weighted_loss',
                        help='Use weighted loss')
    parser.set_defaults(use_weighted_loss=False)
    # Add boolean type command line arguments, use custom str2bool conversion function
    parser.add_argument('--use_mae', type=str2bool, default=True, help="if use mae trick")
    # whether to clamp t
    parser.add_argument('--clamp_t_min', type=float, default=0, help="clamp t")
    # if use unsupervised cfg
    parser.add_argument('--use_unsup_cfg', type=str2bool, default=False, help="if use unsupervised cfg")

    # cfg time interval
    parser.add_argument('--cfg_t_min', type=float, default=0., help="cfg t min")
    parser.add_argument('--cfg_t_max', type=float, default=1., help="cfg t max")

    parser.add_argument('--save_epoch_name', type=str2bool, default=False)

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='images input size')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='vae output embedding dimension')
    parser.add_argument('--vae_stride', default=16, type=int,
                        help='tokenizer stride, default use KL16')
    parser.add_argument('--patch_size', default=1, type=int,
                        help='number of tokens to group as a patch.')
    parser.add_argument('--use_dc_ae', action='store_true', dest='use_dc_ae',
                        help='Use dc ae latents')
    parser.set_defaults(use_dc_ae=False)

    # Generation parameters
    parser.add_argument('--num_iter', default=64, type=int,
                        help='number of autoregressive iterations to generate an image')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='number of images to generate')
    parser.add_argument('--cfg', default=1.0, type=float, help="classifier-free guidance")
    parser.add_argument('--cfg_schedule', default="constant", type=str)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--eval_freq', type=int, default=40, help='evaluation frequency')
    parser.add_argument('--save_last_freq', type=int, default=5, help='save last frequency')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval_bsz', type=int, default=64, help='generation batch size')
    parser.add_argument('--evaluate_multi', action='store_true')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')

    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--ema_rate', default=0.9999, type=float)

    # eMIGM params
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='projection dropout')
    parser.add_argument('--buffer_size', type=int, default=64)

    # Diffusion Loss params
    parser.add_argument('--diffloss_d', type=int, default=12)
    parser.add_argument('--diffloss_w', type=int, default=1536)
    parser.add_argument('--num_sampling_steps', type=str, default="100")
    parser.add_argument('--diffusion_batch_mul', type=int, default=1)
    parser.add_argument('--temperature', default=1.0, type=float, help='diffusion loss sampling temperature')
    # Diffusion Sampling params
    parser.add_argument('--sampling_algo', type=str, default="ddpm", choices=["ddpm", "dpm-solver"], help="sampling algorithm")

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--class_num', default=1000, type=int)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None, required=False,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--resume_epoch', default=None, type=int, help='epoch to resume from')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--distributed_launch", type=str, default="torchrun", choices=["torchrun", "atorch"])

    # caching latents
    parser.add_argument('--use_cached', action='store_true', dest='use_cached',
                        help='Use cached latents')
    parser.set_defaults(use_cached=False)
    parser.add_argument('--cached_path', default='', help='path to cached latents')

    return parser


def main(args):
    if args.distributed_launch == "atorch":
        import atorch
        status = atorch.init_distributed(backend="nccl")
        assert status is True
        dist.init_atorch_distributed_mode(args)
    else:
        misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # augmentation following DiT and ADM
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if args.use_cached:
        dataset_train = CachedFolder(args.cached_path)
    else:
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the vae and emigm model
    if args.use_dc_ae:
        from efficientvit.ae_model_zoo import DCAE_HF
        vae = DCAE_HF.from_pretrained(args.vae_path).cuda().eval()
    else:  # use pretrained VAE
        vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).cuda().eval()

    for param in vae.parameters():
        param.requires_grad = False

    model = emigm.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        label_drop_prob=args.label_drop_prob,
        use_unsup_cfg=args.use_unsup_cfg,
        cfg_t_min=args.cfg_t_min,
        cfg_t_max=args.cfg_t_max,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        mask_strategy=args.mask_strategy,
        use_weighted_loss=args.use_weighted_loss,
        use_mae=args.use_mae,
        clamp_t_min=args.clamp_t_min,
        num_sampling_steps=args.num_sampling_steps,
        sampling_algo=args.sampling_algo,
        diffusion_batch_mul=args.diffusion_batch_mul,
        grad_checkpointing=args.grad_checkpointing,
    )

    print("Model = %s" % str(model))
    # following timm: set wd as 0 for bias and norm layers
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))

    model.to(device)
    model_without_ddp = model


    eff_batch_size = args.batch_size * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # no weight decay on bias, norm layers, and diffloss MLP
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # resume training
    if args.resume:
        # Determine the checkpoint path based on resume_epoch or direct .pth file
        if args.resume.endswith('.pth'):
            checkpoint_path = args.resume
        else:
            checkpoint_path = os.path.join(
                args.resume,
                "checkpoint-last.pth" if args.resume_epoch is None else f"checkpoint-epoch_{args.resume_epoch}.pth"
            )

        # Check if the checkpoint file exists
        if os.path.exists(checkpoint_path):
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check if 'model' exists in checkpoint
            if 'model' in checkpoint:
                model_without_ddp.load_state_dict(checkpoint['model'])
                print(f"Resume checkpoint from {checkpoint_path}")

                # Load EMA parameters if available
                if 'model_ema' in checkpoint:
                    ema_state_dict = checkpoint['model_ema']
                    ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
                else:
                    ema_params = copy.deepcopy(list(model_without_ddp.parameters()))

                # Load optimizer, epoch, and scaler if available
                if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    args.start_epoch = checkpoint['epoch'] + 1
                    if 'scaler' in checkpoint:
                        loss_scaler.load_state_dict(checkpoint['scaler'])
                    print("Resumed optimizer, epoch, and scaler.")
            
            # If 'model' doesn't exist but 'model_ema' does
            elif 'model_ema' in checkpoint:
                model_without_ddp.load_state_dict(checkpoint['model_ema'])
                print(f"Resume checkpoint from {checkpoint_path} using model_ema")
                ema_params = copy.deepcopy(list(model_without_ddp.parameters()))

            # Free up memory
            del checkpoint
        else:
            print(f"Checkpoint file {checkpoint_path} does not exist. Training from scratch.")
            model_params = list(model_without_ddp.parameters())
            ema_params = copy.deepcopy(model_params)
    else:
        # Initialize model and EMA parameters from scratch
        model_params = list(model_without_ddp.parameters())
        ema_params = copy.deepcopy(model_params)
        print("Training from scratch.")

    # evaluate FID and IS
    if args.evaluate:
        torch.cuda.empty_cache()
        if args.evaluate_multi:
            evaluate(model_without_ddp, vae, ema_params, args, 0, batch_size=args.eval_bsz, log_writer=log_writer,
                 cfg=1.0, temperature=args.temperature, use_ema=True)
            evaluate(model_without_ddp, vae, ema_params, args, 0, batch_size=args.eval_bsz // 2, log_writer=log_writer,
                 cfg=args.cfg, temperature=1.0, use_ema=True)
            evaluate(model_without_ddp, vae, ema_params, args, 0, batch_size=args.eval_bsz, log_writer=log_writer,
                 cfg=1.0, temperature=1.0, use_ema=True)
        else:
            evaluate(model_without_ddp, vae, ema_params, args, 0, batch_size=args.eval_bsz, log_writer=log_writer,
                 cfg=args.cfg, temperature=args.temperature, use_ema=True)
        return

    # training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model, vae,
            model_params, ema_params,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # save checkpoint
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            # Save as "last"
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, epoch_name="last")

        # online evaluation
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            if args.save_epoch_name and epoch != 0:
                # Save with epoch number
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, epoch_name=f"epoch_{epoch}")
            torch.cuda.empty_cache()
            if args.evaluate_multi:
                evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=args.eval_bsz, log_writer=log_writer,
                    cfg=1.0, temperature=args.temperature, use_ema=True)
                evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=args.eval_bsz // 2, log_writer=log_writer,
                    cfg=args.cfg, temperature=1.0, use_ema=True)
                evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=args.eval_bsz, log_writer=log_writer,
                    cfg=1.0, temperature=1.0, use_ema=True)
                evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=args.eval_bsz, log_writer=log_writer,
                    cfg=1.0, temperature=args.temperature, use_ema=False)
                evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=args.eval_bsz // 2, log_writer=log_writer,
                    cfg=args.cfg, temperature=1.0, use_ema=False)
                evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=args.eval_bsz, log_writer=log_writer,
                    cfg=1.0, temperature=1.0, use_ema=False)
            else:
                evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=args.eval_bsz, log_writer=log_writer,
                        cfg=1.0, temperature=args.temperature, use_ema=True)
                if not (args.cfg == 1.0 or args.cfg == 0.0):
                    evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=args.eval_bsz // 2,
                            log_writer=log_writer, cfg=args.cfg, temperature=args.temperature, use_ema=True)
            torch.cuda.empty_cache()

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)