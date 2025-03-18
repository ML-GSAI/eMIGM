import datetime
import logging
import os
import socket
import subprocess
import builtins
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
import random 
import numpy as np

logger = logging.getLogger(__name__)

def random_seed(seed=0):
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK") or 0)

def find_free_port(start_port: int, end_port: int):
    """
    Find a free port within the specified range.
    """
    for port in range(start_port, end_port):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", port))  # Try to bind to the port
            s.close()  # Close the socket if successful
            return port
        except OSError as e:
            # print(f"Port {port} is in use, trying next port.")
            continue
    raise RuntimeError(f"No free ports found in range {start_port}-{end_port}")

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print

def init_atorch_distributed_mode(args=SimpleNamespace()):
    #random_seed(getattr(args, "seed", 0)) in eMIGM, seed is setted
    os.environ["RANK"] = str(dist.get_rank())
    os.environ["LOCAL_RANK"] = str(get_local_rank())
    os.environ["WORLD_SIZE"] = str(dist.get_world_size())
    torch.cuda.set_device(get_local_rank())
    
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])

    args.distributed = True
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def init_distributed_mode(args=SimpleNamespace()):
    random_seed(getattr(args, "seed", 0))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.rank = int(os.environ["RANK"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.local_rank = args.gpu
        args.dist_url = "env://"
    elif "SLURM_PROCID" in os.environ:
        os.environ["MASTER_PORT"] = "8966"
        while "MASTER_ADDR" not in os.environ or len(os.environ["MASTER_ADDR"].strip()) == 0:
            os.environ["MASTER_ADDR"] = (
                subprocess.check_output(
                    "sinfo -Nh -n %s | head -n 1 | awk '{print $1}'" % os.environ["SLURM_NODELIST"],
                    shell=True,
                )
                .decode()
                .strip()
            )
            time.sleep(1)
        print(os.environ["MASTER_ADDR"])
        args.world_size = int(os.environ["SLURM_NPROCS"])
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
        args.local_rank = args.gpu
        args.dist_url = "env://"
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        os.environ["RANK"] = str(args.rank)
    else:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(find_free_port(9000, 10000))
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        args.rank = 0
        args.gpu = args.local_rank = 0
        args.world_size = 1
        args.dist_url = "env://"

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}, gpu {}".format(args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(seconds=2 * 60 * 60),
    )
    torch.distributed.barrier()


def all_reduce_mean(x, group=None):
    world_size = dist.get_world_size(group=group)
    if world_size > 1:
        if isinstance(x, torch.Tensor):
            x_reduce = x.clone().cuda()
        else:
            x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce, group=group)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x