import numpy as np
import random
from utils import *
from logger import AverageMeter
import time
from calculate_error import *
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import csv
import os
import imageio
from tqdm import tqdm
from path import Path
import warnings
warnings.filterwarnings(action='ignore')

def validate(args, val_loader, model, decoder, logger, dataset = 'KITTI'):
    ##global device
    batch_time = AverageMeter()
    if dataset == 'KITTI':
        error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3','rmse','rmse_log','log10']
    elif dataset == 'NYU':
        error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'log10', 'a1', 'a2', 'a3','rmse','rmse_log']

    errors = AverageMeter(i=len(error_names))

    # switch to evaluate mode
    model.eval()
    end = time.time()
    logger.valid_bar.update(0)
    for i, (rgb_data, gt_data) in enumerate(val_loader):
        if gt_data.ndim != 4 and gt_data[0] == False:
            continue
        end = time.time()
        rgb_data = rgb_data.cuda()
        gt_data = gt_data.cuda()

        # compute output
        input_img = F.interpolate(rgb_data, scale_factor=0.5, mode='bilinear')
        with torch.no_grad():
            feat_list = model(input_img)
            _,output_depth = decoder(feat_list)
            
            batch_time.update(time.time() - end)
            output_depth = F.interpolate(output_depth, scale_factor=2, mode='bilinear')
            
        if dataset == 'KITTI':
            err_result = compute_errors(gt_data, output_depth,crop=True, cap=args.cap)
        elif dataset == 'NYU':
            err_result = compute_errors_NYU(gt_data, output_depth,crop=True)
        elif dataset == 'Make3D':
            err_result = compute_errors_Make3D(depth, output_depth)

        errors.update(err_result)
        # measure elapsed time
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))

    logger.valid_bar.update(len(val_loader))

    return errors.avg,error_names


def train_single(args,model,decoder, optimizer, dataset_loader,val_loader, batch_size, n_epochs,lr,logger):
    num = 0
    model_num = 0    
      
    encoder_save_dir = './' + args.dataset + '_KDDN_' + args.model + '_b'\
                       + str(args.batch_size) + '_single_encoder'
    decoder_save_dir = './' + args.dataset + '_KDDN_' + args.model + '_b'\
                       + str(args.batch_size) + '_single_decoder'

    if (args.rank == 0):
        print("Training for %d epochs..." % n_epochs)
        if not os.path.exists(encoder_save_dir):
            os.makedirs(encoder_save_dir)
            os.makedirs(decoder_save_dir)

    test_loss_dir = Path(args.save_path)
    test_loss_dir_rmse = str(test_loss_dir/'test_rmse_list.txt')
    test_loss_dir = str(test_loss_dir/'test_loss_list.txt')
    train_loss_dir = Path(args.save_path)
    train_loss_dir_rmse = str(train_loss_dir/'train_rmse_list.txt')
    a1_acc_dir = str(train_loss_dir/'a1_acc_list.txt')
    train_loss_dir = str(train_loss_dir/'train_loss_list.txt')
    loss_pdf = "train_loss.pdf"
    rmse_pdf = "train_rmse.pdf"
    a1_pdf = "train_a1.pdf"        
    
    loss_list = []
    rmse_list = []
    train_loss_list = []
    train_rmse_list = []
    a1_acc_list = []
    num_cnt = 0
    train_loss_cnt = 0

    loss_sum = 0
    n_iter = 0
    iter_per_epoch = len(dataset_loader)
    base_lr = args.lr
    end_lr = args.end_lr
    total_iter = n_epochs * iter_per_epoch
    ################ train mode ####################
    model.train()
    decoder.train()
    ################################################
    for epoch in tqdm(range(n_epochs+5)):
        #dataset_loader.sampler.set_epoch(epoch)
        random.seed(epoch)
        np.random.seed(epoch)               # numpy-based func random setting
        torch.manual_seed(epoch)            # cpu operation random setting
        torch.cuda.manual_seed(epoch)       # gpu operation random setting
        torch.cuda.manual_seed_all(epoch)   # multi-gpu operation random setting
        ####################################### one epoch training #############################################
        for i, (rgb_data, gt_data) in enumerate(dataset_loader):
            # get the inputs
            inputs = rgb_data
            depths = gt_data
            inputs = F.interpolate(inputs, scale_factor=0.5, mode='bilinear')

            inputs = inputs.cuda()
            depths = depths.cuda()

            # wrap them in Variable
            inputs, depths = Variable(inputs), Variable(depths)
            
            '''Network loss'''
            # Feed-forward pass
            feat_list = model(inputs)
            feat_list_, outputs = decoder(feat_list)
            outputs = F.interpolate(outputs, scale_factor=2, mode='bilinear')
            ##################################### Valid mask definition ####################################
            # masking valied area
            valid_mask = make_mask(depths, args.dataset)

            valid_out = outputs[valid_mask]
            valid_gt_sparse = depths[valid_mask]

            ###################################### scale invariant loss #####################################
            scale_inv_loss = scale_invariant_loss(valid_out, valid_gt_sparse)
            #################################################################################################
            
            loss = scale_inv_loss
            
            # zero the parameter gradients and backward & optimize
            optimizer.zero_grad()
            loss.backward()
            if n_iter == total_iter:
                current_lr = end_lr
            else:
                current_lr = (base_lr - end_lr) * (1 - n_iter / total_iter) ** 0.5 + end_lr
                n_iter += 1

            optimizer.param_groups[0]['lr'] = current_lr
            optimizer.param_groups[1]['lr'] = current_lr
            optimizer.step()

            if ((i+1) % 100 == 0):
                if (args.rank == 0):
                    print("epoch: %d,  %d/%d"%(epoch+1,i+1,args.epoch_size))
                    print("[%6d/%6d]  total: %.5f, scale_inv: %.5f"%(n_iter, total_iter, loss.item(),scale_inv_loss.item()))
                    total_loss = loss.item()                    

                    rmse_loss = (torch.sqrt(torch.pow(valid_out-valid_gt_sparse,2))).mean()
                    rmse_loss = rmse_loss.item()
                    
                    train_loss_cnt = train_loss_cnt + 1
                    all_plot(args.save_path,total_loss, rmse_loss, train_loss_list, train_rmse_list, train_loss_dir,train_loss_dir_rmse,loss_pdf, rmse_pdf, train_loss_cnt,True)

        if (args.rank == 0):
            print(" learning decay... current lr: %.6f"%(current_lr))
            torch.save(model.state_dict(), encoder_save_dir+'/epoch_%02d_encoder_loss_%.4f.pkl' %(model_num+1,loss))
            torch.save(decoder.state_dict(), decoder_save_dir+'/epoch_%02d_decoder_loss_%.4f.pkl' %(model_num+1,loss))
        model_num = model_num + 1

    
    return loss

def train_student(args,Teacher,decoder,Student, optimizer, dataset_loader,val_loader, batch_size, n_epochs,lr,logger):
    num = 0
    model_num = 0    
 
    save_dir = './' + args.dataset + '_KDDN_' + args.model + '_b'\
                       + str(args.batch_size)
    
    if (args.rank == 0):
        print("Training for %d epochs..." % n_epochs)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    test_loss_dir = Path(args.save_path)
    test_loss_dir_rmse = str(test_loss_dir/'test_rmse_list.txt')
    test_loss_dir = str(test_loss_dir/'test_loss_list.txt')
    train_loss_dir = Path(args.save_path)
    train_loss_dir_rmse = str(train_loss_dir/'train_rmse_list.txt')
    a1_acc_dir = str(train_loss_dir/'a1_acc_list.txt')
    train_loss_dir = str(train_loss_dir/'train_loss_list.txt')
    loss_pdf = "train_loss.pdf"
    rmse_pdf = "train_rmse.pdf"
    a1_pdf = "train_a1.pdf"        
    
    loss_list = []
    rmse_list = []
    train_loss_list = []
    train_rmse_list = []
    a1_acc_list = []
    num_cnt = 0
    train_loss_cnt = 0

    n_iter = 0
    iter_per_epoch = len(dataset_loader)
    base_lr = args.lr
    end_lr = args.end_lr
    total_iter = n_epochs * iter_per_epoch
    total_iter_drop = (n_epochs-25) * iter_per_epoch
    keep_prob = 1
    ################ train mode ####################
    Student.train()
    Teacher.eval()
    decoder.eval()
    ################################################
    for epoch in tqdm(range(n_epochs+5)):
        dataset_loader.sampler.set_epoch(epoch)
        random.seed(epoch)
        np.random.seed(epoch)               # numpy-based func random setting
        torch.manual_seed(epoch)            # cpu operation random setting
        torch.cuda.manual_seed(epoch)       # gpu operation random setting
        torch.cuda.manual_seed_all(epoch)   # multi-gpu operation random setting

        ####################################### one epoch training #############################################
        for i, (rgb_data, gt_data) in enumerate(dataset_loader):

            # get the inputs
            inputs = rgb_data
            depths = gt_data

            inputs = inputs.cuda()
            inputs = F.interpolate(inputs, scale_factor=0.5, mode='bilinear')
            depths = depths.cuda()

            # wrap them in Variable
            inputs, depths = Variable(inputs), Variable(depths)
            
            '''Network loss'''
            # Feed-forward pass
            with torch.no_grad():
                featlist_T = Teacher(inputs)
                [D_feat1_T, D_feat2_T, D_feat3_T, D_feat4_T, D_feat5_T], outputs_T = decoder(featlist_T)
                D_feat_T_list = [D_feat1_T.detach(), D_feat2_T.detach(), D_feat3_T.detach(), D_feat4_T.detach(), D_feat5_T.detach()]

            featlist = Student(inputs)
            #D_feat_S_list, outputs = decoder(featlist)
            D_feat_S_list, drop_stud_list, drop_tchr_list, outputs = decoder(featlist, keep_prob=keep_prob, T_dec_feat_list=D_feat_T_list, dense_feat_T = featlist_T[-1].detach())

            outputs = F.interpolate(outputs, scale_factor=2, mode='bilinear')
            ##################################### Valid mask definition ####################################
            # masking valied area
            valid_mask = make_mask(depths, args.dataset)

            valid_out = outputs[valid_mask]
            valid_gt_sparse = depths[valid_mask]

            ###################################### scale invariant loss ########################################
            scale_inv_loss = scale_invariant_loss(valid_out, valid_gt_sparse)
            ####################################################################################################
            
            ###################################### feature decomposition loss ##################################
            feat_loss, lap1_loss, lap2_loss, lap3_loss = decomposition_loss(D_feat_S_list, D_feat_T_list)
            feat_loss_ = feat_loss[0] + feat_loss[1] + feat_loss[2] + feat_loss[3] + feat_loss[4]
            lap1_loss_ = lap1_loss[0] + lap1_loss[1] + lap1_loss[2] + lap1_loss[3] + lap1_loss[4]
            lap2_loss_ = lap2_loss[0] + lap2_loss[1] + lap2_loss[2] + lap2_loss[3] + lap2_loss[4]
            lap3_loss_ = lap3_loss[0] + lap3_loss[1] + lap3_loss[2] + lap3_loss[3] + lap3_loss[4]
            ####################################################################################################
            
            ###################################### Dropped feature loss ########################################
            drop_loss = feature_loss_L1(drop_stud_list, drop_tchr_list)
            drop_loss_ = drop_loss[0] + drop_loss[1] + drop_loss[2] + drop_loss[3] + drop_loss[4]
            if n_iter > total_iter_drop:
                keep_prob = 0.8
            else:
                keep_prob = (-0.2/total_iter_drop)*(n_iter) + 1.0
            decoder.module.set_drop_prob(1-keep_prob)
            ####################################################################################################
            
            loss = scale_inv_loss + 0.2*drop_loss_ + 0.3*feat_loss_ + 0.1*lap1_loss_ + 0.1*lap2_loss_ + 0.1*lap3_loss_
            # zero the parameter gradients and backward & optimize
            optimizer.zero_grad()
            loss.backward()
            if n_iter == total_iter:
                current_lr = end_lr
            else:
                current_lr = (base_lr - end_lr) * (1 - n_iter / total_iter) ** 0.5 + end_lr
                n_iter += 1

            optimizer.param_groups[0]['lr'] = current_lr
            optimizer.step()
            
            
            if ((i+1) % 100 == 0):
                if (args.rank == 0):
                    print("epoch: %d,  %d/%d"%(epoch+1,i+1,args.epoch_size))
                    print("[%6d/%6d]  total: %.5f, feat: %.5f, drop: %.5f, lap1_loss: %.5f, lap2_loss: %.5f, lap3_loss: %.5f, scale_inv: %.5f"%(n_iter, total_iter,\
                        loss.item(), 0.3*feat_loss_.item(), 0.2*drop_loss_.item(), 0.1*lap1_loss_.item(), 0.1*lap2_loss_.item(), 0.1*lap3_loss_.item(), scale_inv_loss.item()))
                    
                    print("drop_prob: %.5f"%(1 - keep_prob))
                    total_loss = loss.item()                    
                    
                    rmse_loss = (torch.sqrt(torch.pow(valid_out-valid_gt_sparse,2))).mean()
                    rmse_loss = rmse_loss.item()
                    
                    train_loss_cnt = train_loss_cnt + 1
                    all_plot(args.save_path,total_loss, rmse_loss, train_loss_list, train_rmse_list, train_loss_dir,train_loss_dir_rmse,loss_pdf, rmse_pdf, train_loss_cnt,True)
           
        if (args.rank == 0):
            print(" learning decay... current lr: %.6f"%(current_lr))
            torch.save(Student.state_dict(), save_dir+'/epoch_%02d_encoder_loss_%.4f.pkl' %(model_num+1,loss))
        model_num = model_num + 1
    
    return loss
