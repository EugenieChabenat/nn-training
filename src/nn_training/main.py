import argparse
import os
import random
import time
import warnings
from enum import Enum
import pathlib
import builtins

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

#from nn_training import config, nn_modules
from nn-training.src.nn_training.nn_modules import supervised

#from nn-training.src.nn_training import config, nn_modules

#import os
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

nn_module_names = sorted(name for name in nn_modules.__dict__
    if name.islower() and not name.startswith("__"))

#--added
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/nn-training/src/nn_training/nn_modules')
import supervised, disentangle
#--

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('nn_module', type=str, choices=nn_module_names,
                    help='nn modules: ' +
                        ' | '.join(nn_module_names))
parser.add_argument('--mo', '--model-options', dest='model_options', type=str,
                    help='a string of options passed to nn_module.model (default: "")')
parser.add_argument('--co', '--criterion-options', dest='criterion_options', type=str,
                    help='a string of options passed to nn_module.criterion (default: "")')
parser.add_argument('--oo', '--optimizer-options', dest='optimizer_options', type=str,
                    help='a string of options passed to nn_module.optimizer (default: "")')
parser.add_argument('--so', '--scheduler-options', dest='scheduler_options', type=str,
                    help='a string of options passed to nn_module.scheduler (default: "")')
parser.add_argument('--do', '--dataset-options', dest='dataset_options', type=str,
                    help='a string of options passed to nn_module.dataset (default: "")')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', type=int, metavar='EPOCH',
                    help='resume at the specified epoch (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('-m', '--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-t', '--tensorboard', action='store_true',
                    help='log training statistics to tensorboard')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.world_size == -1:
        if args.dist_url == "env://":
            args.world_size = int(os.environ["WORLD_SIZE"])
        elif 'SLURM_JOB_ID' in os.environ:
            args.world_size = int(os.getenv('SLURM_NNODES'))
    
    if args.rank == -1 and 'SLURM_JOB_ID' in os.environ:
        args.rank = int(os.getenv('SLURM_NODEID'))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
    # suppress printing if not master
    if args.multiprocessing_distributed and args.rank % ngpus_per_node != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    print("=> creating model from nn module '{}'".format(args.nn_module))
    if args.model_options is None:
        model_options = []
    else:
        model_options = args.model_options.split('_')
    model_args = nn_modules.__dict__[args.nn_module].model_parser().parse_args(model_options)
    model = nn_modules.__dict__[args.nn_module].get_model(**vars(model_args))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    if args.criterion_options is None:
        criterion_options = []
    else:
        criterion_options = args.criterion_options.split('_')
    criterion_args = nn_modules.__dict__[args.nn_module].criterion_parser().parse_args(criterion_options)
    criterion = nn_modules.__dict__[args.nn_module].get_criterion(**vars(criterion_args))
    criterion.cuda(args.gpu)

    if args.optimizer_options is None:
        optimizer_options = []
    else:
        optimizer_options = args.optimizer_options.split('_')
    optimizer_args = nn_modules.__dict__[args.nn_module].optimizer_parser().parse_args(optimizer_options)
    optimizer = nn_modules.__dict__[args.nn_module].get_optimizer(model.parameters(), **vars(optimizer_args))
    
    if args.scheduler_options is None:
        scheduler_options = []
    else:
        scheduler_options = args.scheduler_options.split('_')
    scheduler_args = nn_modules.__dict__[args.nn_module].scheduler_parser().parse_args(scheduler_options)
    scheduler = nn_modules.__dict__[args.nn_module].get_scheduler(optimizer, **vars(scheduler_args))
    
    dirname = [args.nn_module, '-b', str(args.batch_size * ngpus_per_node)]
    for name, fullname in [('mo', 'model_options'), ('co', 'criterion_options'), ('oo', 'optimizer_options'), ('so', 'scheduler_options'), ('do', 'dataset_options')]:
        if getattr(args, fullname) is not None:
            dirname += [f'--{name}', getattr(args, fullname)]
    dirname = '_'.join(dirname)
    
    # optionally resume from a checkpoint
    if args.resume is not None:
        args.resume = pathlib.Path(config['paths']['checkpoints']) / dirname / f'checkpoint_{args.resume:04d}.pth.tar'
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.dataset_options is None:
        dataset_options = []
    else:
        dataset_options = args.dataset_options.split('_')
    dataset_args = nn_modules.__dict__[args.nn_module].dataset_parser().parse_args(dataset_options)
    train_dataset = nn_modules.__dict__[args.nn_module].get_dataset(**vars(dataset_args))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # Create SummaryWriter if writing to tensorboard
    args.writer = None
    if args.tensorboard and (not args.multiprocessing_distributed or args.rank % ngpus_per_node == 0):
        args.writer = SummaryWriter(pathlib.Path(config['paths']['tensorboard']) / dirname)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
            
        scheduler.step()

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, dirname=dirname, filename=f'checkpoint_{epoch+1:04d}.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if isinstance(images, torch.Tensor):
            images = [images] # images can be either a tensor or a list of tensor. convert to list of tensors always
            
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = [im.cuda(args.gpu, non_blocking=True) for im in images]
        if torch.cuda.is_available():
            target = {k: v.cuda(args.gpu, non_blocking=True) for k, v in target.items()}

        # compute output
        output = model(*images)
        loss, info = criterion(output, target)

        # record loss
        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
            
            if args.writer is not None:
                stats = {'Loss': loss.item(), 'Time (data)': data_time.val, 'Time (batch)': batch_time.val}
                stats.update({k: v.item() for k, v in info.items()})
                for stat_name, stat in stats.items():
                    args.writer.add_scalar(f'Train/{stat_name}', stat, epoch * len(train_loader) + i + 1)

def save_checkpoint(state, dirname=None, filename='checkpoint.pth.tar'):
    checkpoint_dir = pathlib.Path(config['paths']['checkpoints'])
    if dirname is not None:
        checkpoint_dir = checkpoint_dir / dirname
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_dir / filename)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count])
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
