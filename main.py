import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum


import torch

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Subset
from train import train_epoch
import dataset


from common.renderer_pyrd import Renderer
from common.mocap_dataset import MocapDataset
from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from models.cliff_res50.cliff import CLIFF as cliff_res50
from common import constants


model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
					help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
					choices=model_names,
					help='model architecture: ' +
						' | '.join(model_names) +
						' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
					help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
					help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
					help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
					help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
					help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
					help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
					help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
					help='Use multi-processing distributed training to launch '
						 'N processes per node, which has N GPUs. This is the '
						 'fastest way to use PyTorch for either single node or '
						 'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")

best_acc1 = 0


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
	# torch.save(state, filename)
	if is_best:
		# shutil.copyfile(filename, 'model_best.pth.tar')
		torch.save(state, filename)



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

	if args.dist_url == "env://" and args.world_size == -1:
		args.world_size = int(os.environ["WORLD_SIZE"])

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


# def main_worker(gpu, ngpus_per_node, args):
def main_worker(args):
	global best_acc1
	args.gpu = "cuda:1"
	th = 100000


	# if args.gpu is not None:
	# 	print("Use GPU: {} for training".format(args.gpu))

	# if args.distributed:
	# 	if args.dist_url == "env://" and args.rank == -1:
	# 		args.rank = int(os.environ["RANK"])
	# 	if args.multiprocessing_distributed:
	# 		# For multiprocessing distributed training, rank needs to be the
	# 		# global rank among all the processes
	# 		args.rank = args.rank * ngpus_per_node + gpu
	# 	dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
	# 							world_size=args.world_size, rank=args.rank)

	# Create the model instance
	cliff = eval("cliff_" + "hr48")
	model = cliff(constants.SMPL_MEAN_PARAMS).to(args.gpu)
	# Load the pretrained model
	# state_dict = torch.load("/home/pranoy/Downloads/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt")['model']
	# state_dict = strip_prefix_if_present(state_dict, prefix="module.")
	# cliff_model.load_state_dict(state_dict, strict=True)
	# cliff_model.eval()


	# if not torch.cuda.is_available():
	# 	print('using CPU, this will be slow')
	# elif args.distributed:
	# 	# For multiprocessing distributed, DistributedDataParallel constructor
	# 	# should always set the single device scope, otherwise,
	# 	# DistributedDataParallel will use all available devices.
	# 	if args.gpu is not None:
	# 		torch.cuda.set_device(args.gpu)
	# 		model.cuda(args.gpu)
	# 		# When using a single GPU per process and per
	# 		# DistributedDataParallel, we need to divide the batch size
	# 		# ourselves based on the total number of GPUs of the current node.
	# 		args.batch_size = int(args.batch_size / ngpus_per_node)
	# 		args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
	# 		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
	# 	else:
	# 		model.cuda()
	# 		# DistributedDataParallel will divide and allocate batch_size to all
	# 		# available GPUs if device_ids are not set
	# 		model = torch.nn.parallel.DistributedDataParallel(model)
	# elif args.gpu is not None:
	# 	torch.cuda.set_device(args.gpu)
	# 	model = model.cuda(args.gpu)
	# else:
	# 	# DataParallel will divide and allocate batch_size to all available GPUs
	# 	if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
	# 		model.features = torch.nn.DataParallel(model.features)
	# 		model.cuda()
	# 	else:
	# 		model = torch.nn.DataParallel(model).cuda()
   

	cudnn.benchmark = True

	# args.ngpus_per_node = ngpus_per_node

	# define loss function (criterion), optimizer, and learning rate scheduler
	criterion = nn.MSELoss().cuda(args.gpu)

	# optimizer = torch.optim.SGD(model.parameters(), args.lr,
	# 							momentum=args.momentum,
	# 							weight_decay=args.weight_decay)
	optimizer = torch.optim.Adam(model.parameters(), args.lr)
	#torch.optim.LBFGS(model.parameters(), lr=args.lr, max_iter=1000)
	
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	# scheduler = StepLR(optimizer,30, gamma=0.1)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
	
	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			if args.gpu is None:
				checkpoint = torch.load(args.resume)
			else:
				# Map model to be loaded to specified single gpu.
				loc = 'cuda:{}'.format(args.gpu)
				checkpoint = torch.load(args.resume, map_location=loc)
			args.start_epoch = checkpoint['epoch']
			best_acc1 = checkpoint['best_acc1']
			if args.gpu is not None:
				# best_acc1 may be from a checkpoint from a different GPU
				best_acc1 = best_acc1.to(args.gpu)
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			scheduler.load_state_dict(checkpoint['scheduler'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True


	traindir = "/media/pranoy/Pranoy/mpi_inf_3dhp/S1/Seq1/imageFrames/smpl_params"
	# valdir = os.path.join(args.data, 'val')
	# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
	# 								std=[0.229, 0.224, 0.225])

	train_dataset = dataset.CustomDataset(traindir)

	# val_dataset = datasets.ImageFolder(
	# 	valdir,
	# 	transforms.Compose([
	# 		transforms.Resize(256),
	# 		transforms.CenterCrop(224),
	# 		transforms.ToTensor(),
	# 		normalize,
	# 	]))

	# if args.distributed:
	# 	train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
	# 	# val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
	# else:
	# 	train_sampler = None
	# 	val_sampler = None

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=True,
		num_workers=0, pin_memory=True)

	# val_loader = torch.utils.data.DataLoader(
	# 	val_dataset, batch_size=args.batch_size, shuffle=False,
	# 	num_workers=args.workers, pin_memory=True, sampler=val_sampler)

	# if args.evaluate:
	# 	validate(val_loader, model, criterion, args)
	# 	return

	for epoch in range(args.start_epoch, args.epochs):
		# if args.distributed:
		# 	train_sampler.set_epoch(epoch)

		# train for one epoch
		loss = train_epoch(train_loader, model, criterion, optimizer, epoch, args)

		# # evaluate on validation set
		# acc1 = validate(val_loader, model, criterion, args)
		
		scheduler.step()

		
		# remember best acc@1 and save checkpoint
	
		is_best = loss < th
		th = min(th,loss)
		is_best = True


		# if not args.multiprocessing_distributed or (args.multiprocessing_distributed
		# 		and args.rank % ngpus_per_node == 0):
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': args.arch,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
			'scheduler' : scheduler.state_dict()
		}, is_best)



if __name__ == '__main__':
	args = parser.parse_args()
	main_worker(args)
