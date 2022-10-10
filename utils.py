from __future__ import division
import shutil
import numpy as np
import torch
from path import Path
import datetime
from collections import OrderedDict
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from IPython import display
import itertools
import torch.nn as nn
import os
from torchvision.utils import save_image
import imageio

def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data_path']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path/timestamp

def tensor2array(tensor, max_value=255, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if cv2.__version__.startswith('3'):
                color_cvt = cv2.COLOR_BGR2RGB
            else:  # 2.4
                color_cvt = cv2.cv.CV_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255*tensor.squeeze().numpy()/max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
        array = array.transpose(1,2,0)
    return array

def save_image_tensor(tensor_img,img_dir,filename):
    input_ = tensor_img[0]
    print(tensor_img.shape[1])
    if(tensor_img.shape[1]==1):
        input__ = np.empty([tensor_img.shape[2],tensor_img.shape[3]])
        input__[:,:] = input_[0,:,:]
    elif(tensor_img.shape[1]==3):
        input__ = np.empty([tensor_img.shape[2], tensor_img.shape[3],3])
        input__[:,:,0] = input_[0,:,:]
        input__[:,:,1] = input_[1,:,:]
        input__[:,:,2] = input_[2,:,:]
    else:
        print("file dimension is not proper!!")
        exit()
    plt.imsave(img_dir + '/' + filename, input__, cmap='jet')


def plot_loss(data, apath, epoch,train,filename):
    axis = np.linspace(1, epoch, epoch)
    
    label = 'Total Loss'
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, np.array(data), label=label)
    plt.legend()
    if train is False:
        plt.xlabel('Epochs')
    else:
        plt.xlabel('x100 = Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.savefig(os.path.join(apath, filename))
    plt.close(fig)
    plt.close('all')

def all_plot(save_dir,tot_loss, rmse, loss_list, rmse_list, tot_loss_dir,rmse_dir,loss_pdf, rmse_pdf, count,istrain):
    open_type = 'a' if os.path.exists(tot_loss_dir) else 'w'
    loss_log_file = open(tot_loss_dir, open_type)
    rmse_log_file = open(rmse_dir,open_type)

    loss_list.append(tot_loss)
    rmse_list.append(rmse)

    plot_loss(loss_list, save_dir, count, istrain, loss_pdf)
    plot_loss(rmse_list, save_dir, count, istrain, rmse_pdf)
    loss_log_file.write(('%.5f'%tot_loss) + '\n')
    rmse_log_file.write(('%.5f'%rmse) + '\n')
    loss_log_file.close()
    rmse_log_file.close()

def one_plot(save_dir,tot_loss, loss_list, tot_loss_dir,loss_pdf, count,istrain):
    open_type = 'a' if os.path.exists(tot_loss_dir) else 'w'
    loss_log_file = open(tot_loss_dir, open_type)

    loss_list.append(tot_loss)

    plot_loss(loss_list, save_dir, count, istrain, loss_pdf)
    loss_log_file.write(('%.5f'%tot_loss) + '\n')
    loss_log_file.close()

def BerHu_loss(valid_out, valid_gt):         
    diff = valid_out - valid_gt
    diff_abs = torch.abs(diff)
    c = 0.2*torch.max(diff_abs.detach())         
    mask2 = torch.gt(diff_abs.detach(),c)
    diff_abs[mask2] = (diff_abs[mask2]**2 +(c*c))/(2*c)
    return diff_abs.mean()

def scale_invariant_loss(valid_out, valid_gt):
    logdiff = torch.log(valid_out) - torch.log(valid_gt)
    scale_inv_loss = torch.sqrt((logdiff ** 2).mean() - 0.85*(logdiff.mean() ** 2))*10.0
    return scale_inv_loss

def make_mask(depths, dataset):
    # masking valied area
    if dataset == 'KITTI':
        valid_mask = depths.cpu() > 1
    else:
        valid_mask = depths.cpu() > 0.1
        
    return valid_mask

def similarity_loss(student_feat_list, teacher_feat_list):
    loss = []
    for i in range(len(student_feat_list)):
        S_act = student_feat_list[i]
        T_act = teacher_feat_list[i]
        S_b, S_c, S_h, S_w = S_act.size()
        T_b, T_c, T_h, T_w = S_act.size()
        S_reshape = S_act.view(S_b, S_c*S_h*S_w)
        T_reshape = T_act.view(T_b, T_c*T_h*T_w)
        sim_S = torch.mm(S_reshape, S_reshape.T)
        sim_T = torch.mm(T_reshape, T_reshape.T)
        S_list = []
        T_list = []
        for j in range(S_b):
            norm_factor_S = torch.sum(torch.sqrt(torch.pow(sim_S[j,:],2)))
            norm_factor_T = torch.sum(torch.sqrt(torch.pow(sim_T[j,:],2)))
            sim_S_cat = sim_S[j,:].clone()/norm_factor_S
            sim_T_cat = sim_T[j,:].clone()/norm_factor_T
            S_list.append(sim_S_cat)
            T_list.append(sim_T_cat)
        sim_S = torch.cat(S_list)
        sim_T = torch.cat(T_list)
        loss_ = 100*torch.mean(torch.pow(sim_S-sim_T,2).sum()/(S_b*S_b))
        loss.append(loss_)

    return loss

def feature_loss_L2(student_feat_list, teacher_feat_list):
    loss = []
    for i in range(len(student_feat_list)):
        S_feat = student_feat_list[i]
        T_feat = teacher_feat_list[i]
        diff = S_feat-T_feat
        loss_ = (diff**2).mean()
        loss.append(loss_)

    return loss

def feature_loss_L1(student_feat_list, teacher_feat_list):
    loss = []
    for i in range(len(student_feat_list)):
        S_feat = student_feat_list[i]
        T_feat = teacher_feat_list[i]
        diff = S_feat-T_feat
        loss_ = diff.abs().mean()
        loss.append(loss_)

    return loss

def attention_loss(student_feat_list, teacher_feat_list):
    loss = []
    for i in range(len(student_feat_list)):
        S_feat = student_feat_list[i]
        T_feat = teacher_feat_list[i]
        S_AT_map = S_feat.pow(2).mean(1).view(S_feat.size(0),-1)
        T_AT_map = T_feat.pow(2).mean(1).view(T_feat.size(0),-1)
        S_AT_map_L2norm = F.normalize(S_AT_map)
        T_AT_map_L2norm = F.normalize(T_AT_map)
        loss_ = (S_AT_map_L2norm - T_AT_map_L2norm).pow(2).mean()
        loss.append(1000*loss_)

    return loss

def attention_map_save(input_img,feat_list, index):
    if not os.path.exists('./at_map/'+ str(index)):
        os.makedirs('./at_map/'+ str(index))
    input_img = input_img.cpu().detach()
    for i in range(len(feat_list)):
        feat = feat_list[i]
        AT_map = feat.pow(2).mean(1,keepdim=True)
        AT_map = AT_map.cpu().detach().numpy()
        print(AT_map.shape)
        
        save_image_tensor(AT_map,'./at_map/'+ str(index),'at_%d.png'%i)

def drop_concatenation(student_feat, teacher_feat, keep_prob):
    drop_rate = 1 - keep_prob

    S_feat = student_feat.clone()
    T_feat = teacher_feat.clone()
    n_ch = S_feat.shape[1]
    drop_mask = torch.rand(n_ch) <= drop_rate
    drop_mask = drop_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(S_feat)
    if S_feat[drop_mask].shape[0] != 0:
        drop_S_feat = S_feat[drop_mask].clone()
        drop_T_feat = T_feat[drop_mask].clone()
        S_feat[drop_mask] = (S_feat[drop_mask].abs()/(S_feat[drop_mask].abs()+1e-5))*T_feat[drop_mask]
    else:
        drop_S_feat = torch.tensor(0.).cuda()
        drop_T_feat = torch.tensor(0.).cuda()
        
    return S_feat, drop_S_feat, drop_T_feat

def Laplacian_decomposition(x):
    feat_down2 = F.interpolate(x, scale_factor = 0.5, mode='bilinear')
    feat_down4 = F.interpolate(feat_down2, scale_factor = 0.5, mode='bilinear')
    feat_up2 = F.interpolate(feat_down4, feat_down2.shape[2:], mode='bilinear')
    feat_up = F.interpolate(feat_down2, x.shape[2:], mode='bilinear')
    lap1 = x - feat_up
    lap2 = feat_down2 - feat_up2
    return x, lap1, lap2, feat_down4

def decomposition_loss(student_feat_list, teacher_feat_list):
    feat_loss = []
    lap1_loss = []
    lap2_loss = []
    lap3_loss = []
    alpha_list = [1, 2, 2, 5, 5]
    alpha_list2 = [2, 2, 2, 5, 5]
    for i in range(len(student_feat_list)):
        S_feat = student_feat_list[i]
        T_feat = teacher_feat_list[i]
        alpha = alpha_list[i]
        alpha2 = alpha_list2[i]
        S_feat, S_lap1, S_lap2, S_lap3 = Laplacian_decomposition(S_feat)
        T_feat, T_lap1, T_lap2, T_lap3 = Laplacian_decomposition(T_feat)
        feat_loss_ = ((S_feat-T_feat)**2).mean()
        L1_lap1 = (S_lap1-T_lap1).abs().mean()
        L1_lap2 = (S_lap2-T_lap2).abs().mean()
        L2_lap3 = ((S_lap3-T_lap3)**2).mean()
        feat_loss.append(alpha*feat_loss_)
        lap1_loss.append(alpha2*L1_lap1)
        lap2_loss.append(alpha2*L1_lap2)
        lap3_loss.append(alpha2*L2_lap3)

    return feat_loss, lap1_loss, lap2_loss, lap3_loss





