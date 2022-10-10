from option import args, parser
import csv
import numpy as np
import torch
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn

#################### Distributed learning setting #######################
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
#########################################################################

import torch.optim as optim
import torch.nn as nn
import torch.utils.data

from datasets.datasets_list import MyDataset, Transformer

import os
from itertools import cycle
from utils import *

from logger import AverageMeter

from trainer import validate, train_single, train_student
from model import *

import time

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.multigpu = False
    if args.distributed:
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                 world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size/ngpus_per_node)
        args.workers = int((args.num_workers + ngpus_per_node - 1)/ngpus_per_node)
        print("==> gpu:",args.gpu,", rank:",args.rank,", batch_size:",args.batch_size,", workers:",args.workers)
        torch.cuda.set_device(args.gpu)
    elif args.gpu is None:
        print("==> DataParallel Training")
        args.multigpu = True
        os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_num
    else:
        print("==> Single GPU Training")
        torch.cuda.set_device(args.gpu)

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
        
    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints'/save_path
    if (args.rank == 0):
        print('=> number of GPU: ',args.gpu_num)
        print("=> information will be saved in {}".format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    ######################   Data loading part    ##########################
    if args.dataset == 'KITTI':
        args.max_depth = 80.0
    elif args.dataset == 'NYU':
        args.max_depth = 10.0

    ## define transform function
    train_set = MyDataset(args, train=True)
    test_set = MyDataset(args, train=False)
    if (args.rank == 0):
        print("=> Dataset: ",args.dataset)
        print("=> Data height: {}, width: {} ".format(args.height, args.width))
        print('=> train samples_num: {}  '.format(len(train_set)))
        print('=> test  samples_num: {}  '.format(len(test_set)))

    train_sampler = None
    test_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)
    cudnn.benchmark = True
    ###########################################################################

    ###################### Setting Network, Loss, Optimizer part ###################
    if (args.rank == 0):
        print("=> creating model")
    if args.mode == 'Single_training':
        if (args.rank == 0):
            print('- Single model training')
        assert args.model != "", "Model network is not specified"
        Model_tchr = KDDN_encoder(args, args.model)
        Model_decoder = Basic_decoder_shallow(args, Model_tchr.encoder.out_dimList)
        Model_stdnt = None
    elif args.mode == 'Student_training':
        if (args.rank == 0):
            print('- Student model training')
        assert args.model != "", "Teacher model network is not specified"
        Model_tchr = KDDN_encoder(args, args.T_model)
        Model_decoder = Basic_decoder_shallow(args, Model_tchr.encoder.out_dimList)

        assert args.model != "", "Student model network is not specified"
        Model_stdnt = KDDN_encoder(args, args.model)


    ############################### apex distributed package wrapping ########################
    if args.distributed:
        if args.norm == 'BN':
            if args.mode == 'Student_training':
                Model_tchr = nn.SyncBatchNorm.convert_sync_batchnorm(Model_tchr)
                Model_decoder = nn.SyncBatchNorm.convert_sync_batchnorm(Model_decoder)
                Model_stdnt = nn.SyncBatchNorm.convert_sync_batchnorm(Model_stdnt)
            else:
                Model_stdnt = nn.SyncBatchNorm.convert_sync_batchnorm(Model_stdnt)
            if (args.rank == 0):
                print("=> use SyncBatchNorm")
        Model_tchr = Model_tchr.cuda()
        Model_tchr = DDP(Model_tchr, device_ids=[args.gpu], output_device=args.gpu,
                                                           find_unused_parameters=True)
        Model_decoder = Model_decoder.cuda()
        Model_decoder = DDP(Model_decoder, device_ids=[args.gpu], output_device=args.gpu,
                                                           find_unused_parameters=True)
        if args.mode == 'Student_training':
            Model_stdnt = Model_stdnt.cuda()
            Model_stdnt = DDP(Model_stdnt, device_ids=[args.gpu], output_device=args.gpu,
                                                           find_unused_parameters=True)
        if args.mode == 'Single_training':
            enc_param = Model_tchr.parameters()
            dec_param = Model_decoder.parameters()
        else:
            enc_param = Model_stdnt.parameters()
    elif args.gpu is None:
        if (args.rank == 0):
            print("=> Model Initialized - DataParallel")
        Model_tchr = Model_tchr.cuda()
        Model_tchr = torch.nn.DataParallel(Model_tchr)
        Model_decoder = Model_decoder.cuda()
        Model_decoder = torch.nn.DataParallel(Model_decoder)
        if args.mode == 'Student_training':
            Model_stdnt = Model_stdnt.cuda()
            Model_stdnt = torch.nn.DataParallel(Model_stdnt)
        if args.mode == 'Single_training':
            enc_param = Model_tchr.parameters()
            dec_param = Model_decoder.parameters()
        else:
            enc_param = Model_stdnt.parameters()
    else:
        if (args.rank == 0):
            print("=> Model Initialized on GPU: {} - Single GPU training".format(args.gpu))
        Model_tchr = Model_tchr.cuda()
        Model_decoder = Model_decoder.cuda()
        if args.mode == 'Student_training':
            Model_stdnt = Model_stdnt.cuda()
        if args.mode == 'Single_training':
            enc_param = Model_tchr.parameters()
            dec_param = Model_decoder.parameters()
        else:
            enc_param = Model_stdnt.parameters()
    ###########################################################################################

    ################################ pretrained model loading #################################
    if args.mode == 'Student_training':
        if args.model_encoder_dir != '' and args.model_decoder_dir != '':
            #Model.load_state_dict(torch.load(args.model_dir,map_location='cuda:'+args.gpu_num))
            Model_tchr.load_state_dict(torch.load(args.model_encoder_dir))
            Model_decoder.load_state_dict(torch.load(args.model_decoder_dir))
            if (args.rank == 0):
                print('- pretrained Teacher model is created')
            time.sleep(2)
    #############################################################################################

    ############################# model mode (train/eval) setting ###############################
    if args.mode == 'Single_training':          # Single training
        Model_tchr.train()
        Model_decoder.train()
    elif args.mode == 'Student_training':       # Knowledge distillation
        Model_tchr.eval()
        Model_decoder.eval()
        Model_stdnt.train()

    if args.distributed:
        print("Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("Model Initialized")
    ############################## optimizer and criterion setting ##############################

    if args.mode == 'Single_training':
        optimizer = torch.optim.AdamW([{'params': enc_param, 'weight_decay': args.weight_decay},
                                    {'params': dec_param, 'weight_decay': 0}],
                                    lr=args.lr, eps=args.adam_eps)
    elif args.mode == 'Student_training':
        optimizer = torch.optim.AdamW([{'params': enc_param, 'weight_decay': args.weight_decay}],
                                    lr=args.lr, eps=args.adam_eps)
   
    ##############################################################################################
    logger = None
    
    ################################################################################################

    ####################################### Training part ##########################################
    
    if (args.rank == 0):
        print("training start!")
    if args.mode == 'Single_training':
        loss = train_single(args,Model_tchr, Model_decoder, optimizer, train_loader,
            val_loader,args.batch_size, args.epochs,args.lr,logger)
    elif args.mode == 'Student_training':
        loss = train_student(args,Model_tchr, Model_decoder, Model_stdnt, optimizer, train_loader,
            val_loader,args.batch_size, args.epochs,args.lr,logger)
   
    if (args.rank == 0):
        print("training is finished")


if __name__ == '__main__':
    args.batch_size_dist = args.batch_size
    args.num_threads = args.workers
    args.world_size = 1
    args.rank = 0
    nodes = "127.0.0.1"
    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if args.distributed:
        print("==> Distributed Training")
        mp.set_start_method('forkserver')

        print("==> Initial rank: ",args.rank)
        port = np.random.randint(10000, 10030)
        args.dist_url = 'tcp://{}:{}'.format(nodes, port)
        print("==> dist_url: ",args.dist_url)
        args.dist_backend = 'nccl'
        args.gpu = None
        args.workers = 8
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs = ngpus_per_node, args = (ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)
