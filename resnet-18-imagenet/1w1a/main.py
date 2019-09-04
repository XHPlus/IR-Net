import sys
sys.path.append('./models/modules')
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import argparse
import os
import time
import yaml
import numpy
import logging
from easydict import EasyDict
import pprint
from tensorboardX import SummaryWriter
import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import linklink as link

from models import model_entry
from scheduler import get_scheduler
from memcached_dataset import McDataset
from utils import create_logger, AverageMeter, accuracy, save_checkpoint, load_state, DistributedGivenIterationSampler, simple_group_split, DistributedSampler, param_group_no_wd
from distributed_utils import dist_init, reduce_gradients, DistModule
from loss import LabelSmoothCELoss

from optim import optim_entry, FP16SGD, FusedFP16SGD

#model_names = sorted(name for name in models.__dict__
#    if name.islower() and not name.startswith("__")
#    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--config', default='cfgs/config_res50.yaml')
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--recover', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--sync', action='store_true')
parser.add_argument('--fake', action='store_true')
parser.add_argument('--fuse-prob', action='store_true')
parser.add_argument('--fusion-list', nargs='+', help='multi model fusion list')

parser.add_argument('--arch', default='resnet18')
parser.add_argument('--Tmin', default=0.1, type=float)
parser.add_argument('--Tmax', default=10, type=float)

class ColorAugmentation(object):
    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec == None:
            eig_vec = torch.Tensor([
                [ 0.4009,  0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [ 0.4203, -0.6948, -0.5836],
            ])
        if eig_val == None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val))*0.1
        quatity = torch.mm(self.eig_val*alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor

def main():
    global args, config, best_prec1
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    config = EasyDict(config['common'])
    config.save_path = os.path.dirname(args.config)

    rank, world_size = dist_init()

    # create model
    bn_group_size = config.model.kwargs.bn_group_size
    bn_var_mode = config.model.kwargs.get('bn_var_mode', 'L2')
    if bn_group_size == 1:
        bn_group = None
    else:
        assert world_size % bn_group_size == 0
        bn_group = simple_group_split(world_size, rank, world_size // bn_group_size)

    config.model.kwargs.bn_group = bn_group
    config.model.kwargs.bn_var_mode = (link.syncbnVarMode_t.L1
                                       if bn_var_mode == 'L1'
                                       else link.syncbnVarMode_t.L2)
    model = model_entry(config.model)

    model.cuda()

    if config.optimizer.type == 'FP16SGD' or config.optimizer.type == 'FusedFP16SGD':
        args.fp16 = True
    else:
        args.fp16 = False

    if args.fp16:
        # if you have modules that must use fp32 parameters, and need fp32 input
        # try use link.fp16.register_float_module(your_module)
        # if you only need fp32 parameters set cast_args=False when call this
        # function, then call link.fp16.init() before call model.half()
        if config.optimizer.get('fp16_normal_bn', False):
            print('using normal bn for fp16')
            link.fp16.register_float_module(link.nn.SyncBatchNorm2d, cast_args=False)
            link.fp16.register_float_module(torch.nn.BatchNorm2d, cast_args=False)
            link.fp16.init()
        model.half()

    model = DistModule(model, args.sync)


    # create optimizer
    opt_config = config.optimizer
    opt_config.kwargs.lr = config.lr_scheduler.base_lr
    if config.get('no_wd', False):
        param_group, type2num = param_group_no_wd(model)
        opt_config.kwargs.params = param_group
    else:
        opt_config.kwargs.params = model.parameters()

    optimizer = optim_entry(opt_config)

    # optionally resume from a checkpoint
    last_iter = -1
    best_prec1 = 0
    if args.load_path:
        if args.recover:
            best_prec1, last_iter = load_state(args.load_path, model, optimizer=optimizer)
        else:
            load_state(args.load_path, model)

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # augmentation
    aug = [transforms.RandomResizedCrop(config.augmentation.input_size),
           transforms.RandomHorizontalFlip()]

    for k in config.augmentation.keys():
        assert k in ['input_size', 'test_resize', 'rotation', 'colorjitter', 'colorold']
    rotation = config.augmentation.get('rotation', 0)
    colorjitter = config.augmentation.get('colorjitter', None)
    colorold = config.augmentation.get('colorold', False)

    if rotation > 0:
        aug.append(transforms.RandomRotation(rotation))

    if colorjitter is not None:
        aug.append(transforms.ColorJitter(*colorjitter))

    aug.append(transforms.ToTensor())

    if colorold:
        aug.append(ColorAugmentation())

    aug.append(normalize)

    # train
    train_dataset = McDataset(
        config.train_root,
        config.train_source,
        transforms.Compose(aug),
        fake=args.fake)

    # val
    val_dataset = McDataset(
        config.val_root,
        config.val_source,
        transforms.Compose([
            transforms.Resize(config.augmentation.test_resize),
            transforms.CenterCrop(config.augmentation.input_size),
            transforms.ToTensor(),
            normalize,
        ]),
        args.fake)

    train_sampler = DistributedGivenIterationSampler(train_dataset, config.lr_scheduler.max_iter, config.batch_size,
                                                     last_iter=last_iter)
    val_sampler = DistributedSampler(val_dataset, round_up=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True, sampler=val_sampler, drop_last=True)

    config.lr_scheduler['optimizer'] = optimizer.optimizer if isinstance(optimizer, FP16SGD) else optimizer
    config.lr_scheduler['last_iter'] = last_iter
    lr_scheduler = get_scheduler(config.lr_scheduler)

    if rank == 0:
        tb_logger = SummaryWriter(config.save_path+'/events')
        logger = create_logger('global_logger', config.save_path+'/log.txt')
        logger.info('args: {}'.format(pprint.pformat(args)))
        logger.info('config: {}'.format(pprint.pformat(config)))
    else:
        tb_logger = None

    if args.evaluate:
        if args.fusion_list is not None:
            validate(val_loader, model, fusion_list=args.fusion_list, fuse_prob=args.fuse_prob)
        else:
            validate(val_loader, model)
        link.finalize()
        return

    train(train_loader, val_loader, model, optimizer, lr_scheduler, last_iter+1, tb_logger)

    link.finalize()

def train(train_loader, val_loader, model, optimizer, lr_scheduler, start_iter, tb_logger):

    global best_prec1

    batch_time = AverageMeter(config.print_freq)
    fw_time = AverageMeter(config.print_freq)
    bp_time = AverageMeter(config.print_freq)
    sy_time = AverageMeter(config.print_freq)
    step_time = AverageMeter(config.print_freq)
    data_time = AverageMeter(config.print_freq)
    losses = AverageMeter(config.print_freq)
    top1 = AverageMeter(config.print_freq)
    top5 = AverageMeter(config.print_freq)

    # switch to train mode
    model.train()

    world_size = link.get_world_size()
    rank = link.get_rank()

    logger = logging.getLogger('global_logger')

    end = time.time()

    label_smooth = config.get('label_smooth', 0.0)
    if label_smooth > 0:
        logger.info('using label_smooth: {}'.format(label_smooth))
        criterion = LabelSmoothCELoss(label_smooth, 1000)
    else:
        criterion = nn.CrossEntropyLoss()
        
    T_min, T_max = args.Tmin, args.Tmax
    # print (T_min, T_max)
        
    def Log_UP(K_min, K_max, ITEMS, ALL_ITEMS):
        Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
        return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / ALL_ITEMS * ITEMS)]).float().cuda()
    
    # print (model)
    TIME = time.time()

    for i, (input, target) in enumerate(train_loader):
        
        curr_step = start_iter + i
        lr_scheduler.step(curr_step)
        current_lr = lr_scheduler.get_lr()[0]
    
        if (curr_step % config.print_freq == 0):
            t = Log_UP(T_min, T_max, curr_step, len(train_loader))
            if (t < 1):
                k = 1 / t
            else:
                k = torch.tensor([1]).float().cuda()
            
            for i in range(2):
                
                model.module.layer1[i].conv1.k = k
                model.module.layer1[i].conv2.k = k
                model.module.layer1[i].conv1.t = t
                model.module.layer1[i].conv2.t = t
                
                model.module.layer2[i].conv1.k = k
                model.module.layer2[i].conv2.k = k
                model.module.layer2[i].conv1.t = t
                model.module.layer2[i].conv2.t = t
                
                model.module.layer3[i].conv1.k = k
                model.module.layer3[i].conv2.k = k
                model.module.layer3[i].conv1.t = t
                model.module.layer3[i].conv2.t = t
                
                model.module.layer4[i].conv1.k = k
                model.module.layer4[i].conv2.k = k
                model.module.layer4[i].conv1.t = t
                model.module.layer4[i].conv2.t = t

            # print ('current k {:.5e} current t {:.5e}'.format(k[0], t[0]))

        # measure data loading time
        data_time.update(time.time() - end)

        # transfer input to gpu
        target = target.cuda()
        input = input.cuda() if not args.fp16 else input.cuda().half()

        # forward
        output = model(input)
        loss = criterion(output, target) / world_size

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        reduced_loss = loss.clone()
        reduced_prec1 = prec1.clone() / world_size
        reduced_prec5 = prec5.clone() / world_size

        link.allreduce(reduced_loss)
        link.allreduce(reduced_prec1)
        link.allreduce(reduced_prec5)

        losses.update(reduced_loss.item())
        top1.update(reduced_prec1.item())
        top5.update(reduced_prec5.item())

        # backward
        optimizer.zero_grad()

        if isinstance(optimizer, FusedFP16SGD):
            optimizer.backward(loss)
            reduce_gradients(model, args.sync)
            optimizer.step()
        elif isinstance(optimizer, FP16SGD):
            def closure():
                # backward
                optimizer.backward(loss, False)
                # sync gradients
                reduce_gradients(model, args.sync)
                # check overflow, convert to fp32 grads, downscale
                optimizer.update_master_grads()
                return loss
            optimizer.step(closure)
        else:
            loss.backward()
            reduce_gradients(model, args.sync)
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if curr_step % config.print_freq == 0 and rank == 0:
            tb_logger.add_scalar('loss_train', losses.avg, curr_step)
            tb_logger.add_scalar('acc1_train', top1.avg, curr_step)
            tb_logger.add_scalar('acc5_train', top5.avg, curr_step)
            tb_logger.add_scalar('lr', current_lr, curr_step)
            logger.info('Iter: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'LR {lr:.4f}'.format(
                   curr_step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=current_lr))

        if curr_step > 0 and curr_step % config.val_freq == 0:
            val_loss, prec1, prec5 = validate(val_loader, model)

            if not tb_logger is None:
                tb_logger.add_scalar('loss_val', val_loss, curr_step)
                tb_logger.add_scalar('acc1_val', prec1, curr_step)
                tb_logger.add_scalar('acc5_val', prec5, curr_step)


            if rank == 0:
                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint({
                    'step': curr_step,
                    'arch': config.model.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, config.save_path+'/ckpt'+str(TIME % 100000))

        end = time.time()

def validate(val_loader, model, fusion_list=None, fuse_prob=False):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)
    top1 = AverageMeter(0)
    top5 = AverageMeter(0)

    # switch to evaluate mode
    if fusion_list is not None:
        model_list = []
        for i in range(len(fusion_list)):
            model_list.append(model_entry(config.model))
            model_list[i].cuda()
            model_list[i] = DistModule(model_list[i], args.sync)
            load_state(fusion_list[i], model_list[i])
            model_list[i].eval()
        if fuse_prob:
            softmax = nn.Softmax(dim=1)
    else:
        model.eval()

    rank = link.get_rank()
    world_size = link.get_world_size()

    logger = logging.getLogger('global_logger')

    criterion = nn.CrossEntropyLoss()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda() if not args.fp16 else input.half().cuda()
            target = target.cuda()
            # compute output
            if fusion_list is not None:
                output_list = []
                for model_idx in range(len(fusion_list)):
                    output = model_list[model_idx](input)
                    if fuse_prob:
                        output = softmax(output)
                    output_list.append(output)
                output = torch.stack(output_list, 0)
                output = torch.mean(output, 0)
            else:
                output = model(input)

            # measure accuracy and record loss
            loss = criterion(output, target) #/ world_size ## loss should not be scaled here, it's reduced later!
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            num = input.size(0)
            losses.update(loss.item(), num)
            top1.update(prec1.item(), num)
            top5.update(prec5.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0 and rank == 0:
                logger.info('Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time))

    # gather final results
    total_num = torch.Tensor([losses.count])
    loss_sum = torch.Tensor([losses.avg*losses.count])
    top1_sum = torch.Tensor([top1.avg*top1.count])
    top5_sum = torch.Tensor([top5.avg*top5.count])
    link.allreduce(total_num)
    link.allreduce(loss_sum)
    link.allreduce(top1_sum)
    link.allreduce(top5_sum)
    final_loss = loss_sum.item()/total_num.item()
    final_top1 = top1_sum.item()/total_num.item()
    final_top5 = top5_sum.item()/total_num.item()

    if rank == 0:
        logger.info(' * Prec@1 {:.3f}\tPrec@5 {:.3f}\tLoss {:.3f}\ttotal_num={}'.format(final_top1, final_top5, final_loss, total_num.item()))

    model.train()

    return final_loss, final_top1, final_top5

if __name__ == '__main__':
    main()
