# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Email:  alisonzli@tencent.com or alisonbrielee@gmail.com
# **********************************
import sys
import torch.nn.functional as F
import torch.nn as nn
from layers3D import SpatialTransformer, ResizeTransform 
from utils.utils import *
import torch
from torch import Tensor
import math
import nibabel as nib
from scipy import ndimage
import GeodisTK
import utils.utils as utils

# shape = (192, 160)
# shape = (160, 192, 224)
# shape = (160, 160, 160)
shape = (128, 128, 128)

class mask_cross_entropy_bpm(nn.Module):
    def __init__(self):
        super(mask_cross_entropy_bpm, self).__init__()
    def forward(self, pred, target):
        batch_num = pred.size()[0]
        labels = pred.size()[1]
        inds = torch.arange(0, batch_num, dtype=torch.long, device=pred.device)
        focal_weight = torch.zeros((pred.size()), dtype=torch.float32, device=pred.device)
        pred_sm = F.softmax(pred, dim=1).float()
        for label in range(0, labels):
            pred_slice = pred_sm[inds, label].unsqueeze(1)          
            w = 1
            rp_padding = torch.nn.ReflectionPad3d(w)
            p = rp_padding(pred_slice)
            p = torch.clamp(p, min=1/labels)

            # laplacian_kernel_1 3D 中间值为 27 
            laplacian_kernel_1 = torch.zeros((2*w+1, 2*w+1, 2*w+1), dtype=torch.float32, device=pred.device).reshape(1,1,2*w+1,2*w+1,2*w+1).requires_grad_(False) - 1   
            laplacian_kernel_1[0,0,w,w,w] = (2*w+1)*(2*w+1)*(2*w+1) - 1
            
            boundary_target = F.conv3d(p, laplacian_kernel_1)
            boundary_target = torch.abs(boundary_target)
            focal_weight[inds, label,:] = boundary_target.detach()
            # print('a', focal_weight.shape)

        # focal_weight_avg = torch.mean(focal_weight, 1).unsqueeze(1)
        focal_weight = 0.5 / (focal_weight + 0.5)

        # print('b', pred.shape)
        ce_loss = F.cross_entropy(pred*focal_weight, target, reduction='none')
        # ce_loss = ce_loss * focal_weight_avg
        ce_loss = ce_loss.mean()
        return ce_loss

class LossFunction_dice_2GT(nn.Module):
    def __init__(self):
        super(LossFunction_dice_2GT, self).__init__()
        self.dice_loss = Dice()
        self.spatial_transform = SpatialTransformer(volsize=shape)

    def forward(self, mask_0, mask_1, flow):
        mask_0 = F.one_hot(mask_0.squeeze(1).to(torch.int64), num_classes=14).permute(0, 4, 1, 2, 3).float()
        mask_1 = F.one_hot(mask_1.squeeze(1).to(torch.int64), num_classes=14).permute(0, 4, 1, 2, 3).float()
        mask = self.spatial_transform(mask_1, flow)
        loss = self.dice_loss(mask_0, mask)
        return loss

class LossFunction_dice(nn.Module):
    def __init__(self):
        super(LossFunction_dice, self).__init__()
        # GT在前
        self.dice_loss = Dice()
        self.spatial_transform = SpatialTransformer()

    def forward(self, mask_0, mask_1, flow, num_classes):
        mask_1 = F.one_hot(mask_1.squeeze(1).to(torch.int64), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        mask = self.spatial_transform(mask_1, flow)
        loss = self.dice_loss(mask, mask_0)
        return loss


class LossFunction_dice_ce(nn.Module):
    def __init__(self):
        super(LossFunction_dice_ce, self).__init__()
        # GT在前
        self.dice_loss = Dice()
        self.ce = cross_entropy()
        self.spatial_transform = SpatialTransformer()

    def forward(self, mask_0, mask_1, flow, wapred_mask_1):
        mask_1 = F.one_hot(mask_1.squeeze(1).to(torch.int64), num_classes=14).permute(0, 4, 1, 2, 3).float()
        mask = self.spatial_transform(mask_1, flow)
        dice = self.dice_loss(mask, F.softmax(mask_0, dim=1).float())
        ce = self.ce(mask_0, wapred_mask_1) # (14, 128, 128, 128) (128, 128, 128)
        loss = dice + ce
        return loss, dice, ce


class LossFunction_Segmentation(nn.Module):
    def __init__(self):
        super(LossFunction_Segmentation, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = MulticlassDice()

    def forward(self, mask_pred, mask_GT, n_classes=14):
        ce = self.ce_loss(mask_pred, mask_GT.squeeze(1).long())  # (14, 30, 192, 160) (14, 192, 160)
        dice = self.dice_loss(F.softmax(mask_pred, dim=1).float(),
                              F.one_hot(mask_GT.squeeze(1).to(torch.int64), n_classes).permute(0, 3, 1, 2).float())
        loss = ce + dice
        return loss, ce, dice


###  111
class LossFunctionNcc_Registration(nn.Module):
    def __init__(self):
        super(LossFunctionNcc_Registration, self).__init__()
        self.gradient_loss = gradient_loss()

    def forward(self, y, tgt, src, flow):
        hyper_1 = 10
        hyper_2 = 15
        ncc = nas_ncc(tgt, y)
        grad = self.gradient_loss(flow)

        loss = hyper_1 * ncc + hyper_2 * grad
        return loss, ncc, grad

class LossFunction_FixedEncoder_Reg_KL(nn.Module):
    def __init__(self):
        super(LossFunction_FixedEncoder_Reg_KL, self).__init__()
        self.gradient_loss = gradient_loss()
        self.ce_loss = cross_entropy()
        self.kl = torch.nn.KLDivLoss(reduction='mean')

    def forward(self, y, tgt, flow, warped_x_logits_sm, tgt_logits_sm):
        hyper_1 = 10
        hyper_2 = 15
        hyper_3 = 20
        ncc = nas_ncc(tgt, y)
        grad = self.gradient_loss(flow)

        # ce = self.ce_loss(tgt_logits, warped_x_seg)  # (16, 30, 192, 160) (16, 192, 160)
        kl = self.kl(warped_x_logits_sm, tgt_logits_sm)

        loss = hyper_1 * ncc + hyper_2 * grad + hyper_3 * kl

        return loss, ncc, grad, kl

class LossFunction_FixedEncoder_Reg_Dice(nn.Module):
    def __init__(self):
        super(LossFunction_FixedEncoder_Reg_Dice, self).__init__()
        self.gradient_loss = gradient_loss()
        self.ce_loss = cross_entropy()
        self.dice_loss = Dice()
        self.spatial_transform = SpatialTransformer()

    def forward(self, y, tgt, flow, src_logits, tgt_logits):
        hyper_1 = 10
        hyper_2 = 15
        hyper_3 = 1
        ncc = nas_ncc(tgt, y)
        grad = self.gradient_loss(flow)

        # ce = self.ce_loss(tgt_logits, warped_x_seg)  # (16, 30, 192, 160) (16, 192, 160)
        warped_src_logits = self.spatial_transform(src_logits, flow)
        dice = self.dice_loss(F.softmax(warped_src_logits, dim=1).float(), F.softmax(tgt_logits, dim=1).float())

        loss = hyper_1 * ncc + hyper_2 * grad + hyper_3 * dice

        return loss, ncc, grad, dice

class LossFunctionNccIntensity_Registration(nn.Module):
    def __init__(self):
        super(LossFunctionNccIntensity_Registration, self).__init__()
        self.gradient_loss = gradient_loss()
        self.mse = nn.MSELoss()

    def forward(self, intensity, warped_src, tgt, src, flow):
        hyper_1 = 10
        hyper_2 = 15
        hyper_3 = 10
        ncc = nas_ncc(tgt, warped_src)
        grad = self.gradient_loss(flow)
        mse = hyper_3 * self.mse(tgt, intensity)

        warp_loss = hyper_1 * ncc + hyper_2 * grad
        total_loss = warp_loss + mse
        return total_loss, warp_loss, ncc, grad, mse

class mask_dice_bpm_inverse(nn.Module):
    def __init__(self):
        super(mask_dice_bpm_inverse, self).__init__()
        self.dice_loss = Dice_bpm()
    def forward(self, pred, target):
        batch_num = pred.size()[0]
        labels = pred.size()[1]
        inds = torch.arange(0, batch_num, dtype=torch.long, device=pred.device)
        focal_weight = torch.zeros((pred.size()), dtype=torch.float32, device=pred.device)
        pred_sm = F.softmax(pred, dim=1).float()
        for label in range(0, labels):
            pred_slice = pred_sm[inds, label].unsqueeze(1)          
            w = 1
            rp_padding = torch.nn.ReflectionPad3d(w)
            p = rp_padding(pred_slice)
            p = torch.clamp(p, min=0.071)

            # laplacian_kernel_1 3D 中间值为 27 
            laplacian_kernel_1 = torch.zeros((2*w+1, 2*w+1, 2*w+1), dtype=torch.float32, device=pred.device).reshape(1,1,2*w+1,2*w+1,2*w+1).requires_grad_(False) - 1   
            laplacian_kernel_1[0,0,w,w,w] = (2*w+1)*(2*w+1)*(2*w+1) - 1
            
            # laplacian_kernel_2 = np.array([[[0,0,0], [0,-1,0], [0,0,0]],\
            #                            [[0,-1,0],[-1,7,-1],[0,-1,0]],\
            #                            [[0,0,0], [0,-1,0], [0,0,0]]])
            # laplacian_kernel_2 = torch.as_tensor(laplacian_kernel_2, dtype=torch.float32, device=pred.device).reshape(1,1,2*w+1,2*w+1,2*w+1)
        
            boundary_target = F.conv3d(p, laplacian_kernel_1)
            boundary_target = torch.abs(boundary_target)
            # save_nii(boundary_target, '/temp4/Reg-Seg/Figure/boundary/_boundary_max-_{}.nii'.format(label))
            # boundary_target = boundary_target.squeeze(1)
            # boundary_target = boundary_target / torch.mean(torch.mean(torch.mean(boundary_target, -1), -1), -1)
            focal_weight[inds, label,:] = boundary_target.detach()
            # save_nii(boundary_target, '/temp4/Reg-Seg/Figure/boundary/boundary_{}.nii'.format(label))

        # focal_weight_avg = torch.mean(focal_weight, 1).unsqueeze(1)
        # save_nii(focal_weight_avg, '/temp4/Reg-Seg/Figure/boundary/1/seg/30_boundary_avg.nii')
        focal_weight = 0.5 / (focal_weight + 0.5)
        # save_nii(focal_weight_avg, '/temp4/Reg-Seg/Figure/boundary/1/seg/30_boundary_reverse_avg.nii')
        # import pdb; pdb.set_trace()
        dice_loss = self.dice_loss(pred, target, focal_weight)
        # import pdb; pdb.set_trace()
        return dice_loss

class LossFunction_RegLoG(nn.Module):
    def __init__(self):
        super(LossFunction_RegLoG, self).__init__()
        self.gradient_loss = gradient_loss()

    def forward(self, src, warped_src, tgt, flow, int_flow1, int_flow2, seg_pred):        
        hyper_1 = 5
        hyper_2 = 15
        hyper_3 = 10

        lncc_loss = nas_ncc(tgt, warped_src)
        grad_loss = self.gradient_loss(flow)
        boundary_ncc = LoG_ncc(tgt, warped_src, seg_pred, 14)
        warp_loss = hyper_1 * lncc_loss +  hyper_2 * grad_loss + hyper_3 * boundary_ncc

        return warp_loss, boundary_ncc

def LoG_ncc(I, J, pred, labels): 
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    win = [9] * ndims
    sum_filt = torch.ones([1, 1, *win]).to("cuda")
    pad_no = math.floor(win[0] / 2)
    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)
        
    pred = F.softmax(pred, dim=1).float()
    weight = LoG(pred)
    # weight_ = LoG_sm(pred)
    # import pdb; pdb.set_trace()
    
    # I_var, J_var, cross = compute_local_sums(weight*I, weight*J, sum_filt, stride, padding, win)
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    
    ncc = weight * cc

    return -1 * torch.mean(ncc)

def LoG(pred):
    batch_num = pred.size()[0]
    labels = pred.size()[1]
    inds = torch.arange(0, batch_num, dtype=torch.long, device=pred.device)
    focal_weight = torch.zeros((pred.size()), dtype=torch.float32, device=pred.device)
    for label in range(0, labels):
        # import pdb; pdb.set_trace()
        pred_slice = pred[inds, label].unsqueeze(1)
        # pred_softmax = F.softmax(pred, dim=1).float()
        # pred_sigmoid = pred_slice.sigmoid()
        # target = target.type_as(pred) 

        # p = pred_sigmoid.unsqueeze(1)
        
        w = 1
        rp_padding = torch.nn.ReflectionPad3d(w)
        p = rp_padding(pred_slice)
        p = torch.clamp(p, min=0.071)
        # save_nii(p, '/temp4/Reg-Seg/Figure/boundary/prob_{}.nii'.format(label))
        

        # laplacian_kernel_1 3D 中间值为 27 
        laplacian_kernel_1 = torch.zeros((2*w+1, 2*w+1, 2*w+1), dtype=torch.float32, device=pred.device).reshape(1,1,2*w+1,2*w+1,2*w+1).requires_grad_(False) - 1   
        laplacian_kernel_1[0,0,w,w,w] = (2*w+1)*(2*w+1)*(2*w+1) - 1
        
        # laplacian_kernel_2 3D 中间值为 7
        # laplacian_kernel_2 = np.array([[[0,0,0], [0,-1,0], [0,0,0]],\
        #                                [[0,-1,0],[-1,7,-1],[0,-1,0]],\
        #                                [[0,0,0], [0,-1,0], [0,0,0]]])
        # laplacian_kernel_2 = torch.as_tensor(laplacian_kernel_2, dtype=torch.float32, device=pred.device).reshape(1,1,2*w+1,2*w+1,2*w+1)

        # laplacian_kernel_gaussian
        # boundary_target = ndimage.gaussian_laplace(p.cpu().detach().numpy().squeeze(), sigma=1)
        # boundary_target = torch.as_tensor(boundary_target, dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(1)
        # save_nii(boundary_target, '/temp4/Reg-Seg/Figure/p_{}.nii'.format(label))
       
        boundary_target = F.conv3d(p, laplacian_kernel_1)
        boundary_target = torch.abs(boundary_target)
        # save_nii(boundary_target, '/temp4/Reg-Seg/Figure/boundary/conv_abs_boundary_{}.nii'.format(label))
        # boundary_target = boundary_target.squeeze(1)
        # boundary_target = boundary_target / torch.mean(torch.mean(torch.mean(boundary_target, -1), -1), -1)
        focal_weight[inds, label,:] = boundary_target.detach()
        # save_nii(boundary_target, '/temp4/Reg-Seg/Figure/boundary/boundary_{}.nii'.format(label))
    # focal_weight_sum = torch.sum(focal_weight, 1)
    # import pdb; pdb.set_trace()
    focal_weight_avg = torch.sum(focal_weight, 1).unsqueeze(1)
    # save_nii(focal_weight_avg, '/temp4/Reg-Seg/Figure/boundary/1/reg/30_boundary_avg.nii')

    # import pdb; pdb.set_trace()
    # save_nii(focal_weight_sum, '/temp4/Reg-Seg/Figure/boundary_sum.nii')
    # save_nii(vol, '/temp4/Reg-Seg/Figure/vol.nii')
    # save_nii(target, '/temp4/Reg-Seg/Figure/target.nii')
    # a_max = torch.max(focal_weight_avg)
    # a_min = torch.min(focal_weight_avg)

    # image_boundary = F.conv3d(vol, laplacian_kernel_1, padding=w)
    # image_boundary = torch.abs(image_boundary)
    # image_boundary = image_boundary.squeeze(1)
    # image_boundary = image_boundary / torch.mean(torch.mean(torch.mean(image_boundary, -1), -1), -1)
    # save_nii(image_boundary, '/temp4/Reg-Seg/Figure/image_boundary.nii')

    # seg_boundary = F.conv3d(target, laplacian_kernel_1, padding=w)
    # seg_boundary = torch.abs(seg_boundary)
    # seg_boundary = seg_boundary.squeeze(1)
    # seg_boundary = seg_boundary / torch.mean(torch.mean(torch.mean(seg_boundary, -1), -1), -1)
    # save_nii(seg_boundary, '/temp4/Reg-Seg/Figure/seg_boundary.nii')
    # import pdb; pdb.set_trace()
    return focal_weight_avg  

class Dice_bpm(nn.Module):
    """
    N-D dice for segmentation
    """
    def __init__(self):
        super(Dice_bpm, self).__init__()

    def forward(self, y_pred, y_true, bpm):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred * bpm).sum(dim=vol_axes)
        bottom = torch.clamp(((y_true + y_pred)* bpm).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice

class gradient_loss(nn.Module):
    def __init__(self):
        super(gradient_loss, self).__init__()

    def forward(self, s, penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
        dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
        dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])
        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0

class multi_loss(nn.Module):
    def __init__(self):
        super(multi_loss, self).__init__()

        inshape = shape
        down_shape2 = [int(d / 4) for d in inshape]
        down_shape1 = [int(d / 2) for d in inshape]
        self.spatial_transform_1 = SpatialTransformer(volsize=down_shape1)
        self.spatial_transform_2 = SpatialTransformer(volsize=down_shape2)
        self.resize_1 = ResizeTransform(2, len(inshape))
        self.resize_2 = ResizeTransform(4, len(inshape))

    def forward(self, src, tgt, flow1, flow2, hyper_3, hyper_4):
        loss = 0.
        zoomed_x1 = self.resize_1(tgt)
        zoomed_x2 = self.resize_1(src)
        warped_zoomed_x2 = self.spatial_transform_1(zoomed_x2, flow1)
        loss += hyper_3 * similarity_loss(warped_zoomed_x2, zoomed_x1)

        zoomed_x1 = self.resize_2(tgt)
        zoomed_x2 = self.resize_2(src)
        warped_zoomed_x2 = self.spatial_transform_2(zoomed_x2, flow2)
        loss += hyper_4 * similarity_loss(warped_zoomed_x2, zoomed_x1)

        return loss

class multi_loss_mask(nn.Module):
    def __init__(self):
        super(multi_loss_mask, self).__init__()

        inshape = shape
        down_shape2 = [int(d / 4) for d in inshape]
        down_shape1 = [int(d / 2) for d in inshape]
        self.spatial_transform_1 = SpatialTransformer(volsize=down_shape1, mode='nearest')
        self.spatial_transform_2 = SpatialTransformer(volsize=down_shape2, mode='nearest')
        self.resize_1 = ResizeTransform(2, len(inshape))
        self.resize_2 = ResizeTransform(4, len(inshape))
        self.sim_loss = MulticlassDice()

    def forward(self, src, tgt, flow1, flow2, hyper_3, hyper_4):
        loss = 0.
        zoomed_x1 = self.resize_1(tgt)
        zoomed_x2 = self.resize_1(src)
        warped_zoomed_x2 = self.spatial_transform_1(zoomed_x2, flow1)
        loss += hyper_3 * self.sim_loss(F.one_hot(warped_zoomed_x2[:, 0, :, :].to(torch.int64), 14).permute(0, 3, 1, 2).float(),
                                        F.one_hot(zoomed_x1[:, 0, :, :].to(torch.int64), 14).permute(0, 3, 1, 2).float())

        zoomed_x1 = self.resize_2(tgt)
        zoomed_x2 = self.resize_2(src)
        warped_zoomed_x2 = self.spatial_transform_2(zoomed_x2, flow2)
        loss += hyper_4 * self.sim_loss(F.one_hot(warped_zoomed_x2[:, 0, :, :].to(torch.int64), 14).permute(0, 3, 1, 2).float(),
                                        F.one_hot(zoomed_x1[:, 0, :, :].to(torch.int64), 14).permute(0, 3, 1, 2).float())

        return loss

class MulticlassDice(nn.Module):
    def __init__(self):
        super(MulticlassDice, self).__init__()

    def forward(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
        # Dice loss (objective to minimize) between 0 and 1
        # Average of Dice coefficient for all classes
        assert input.size() == target.size()
        dice = 0
        for channel in range(input.shape[1]):
            dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
        return 1 - dice / input.shape[1]

class LossFunction_RegLoG(nn.Module):
    def __init__(self):
        super(LossFunction_RegLoG, self).__init__()
        self.gradient_loss = gradient_loss()

    def forward(self, src, warped_src, tgt, flow, seg_pred):  
        hyper_1 = 5
        hyper_2 = 15
        hyper_3 = 10

        lncc_loss = nas_ncc(tgt, warped_src)
        grad_loss = self.gradient_loss(flow)
        boundary_ncc = LoG_ncc(tgt, warped_src, seg_pred, 14)
        warp_loss = hyper_1 * lncc_loss  + hyper_2 * grad_loss + hyper_3 * boundary_ncc
        return warp_loss, boundary_ncc

def similarity_loss(tgt, warped_img):
    sizes = np.prod(list(tgt.shape)[1:])
    flatten1 = torch.reshape(tgt, (-1, sizes))
    flatten2 = torch.reshape(warped_img, (-1, sizes))

    mean1 = torch.reshape(torch.mean(flatten1, dim=-1), (-1, 1))
    mean2 = torch.reshape(torch.mean(flatten2, dim=-1), (-1, 1))
    var1 = torch.mean((flatten1 - mean1) ** 2, dim=-1)
    var2 = torch.mean((flatten2 - mean2) ** 2, dim=-1)
    cov12 = torch.mean((flatten1 - mean1) * (flatten2 - mean2), dim=-1)
    pearson_r = cov12 / torch.sqrt((var1 + 1e-6) * (var2 + 1e-6))
    raw_loss = torch.sum(1 - pearson_r)

    return raw_loss

class cross_entropy(nn.Module):
    def __init__(self):
        super(cross_entropy, self).__init__()

    def forward(self, input, target, weight=None, size_average=True):
        n, c, h, w, s = input.size()
        log_p = F.log_softmax(input, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
        target = target.view(target.numel())
        loss = F.nll_loss(log_p, target.long(), weight=weight, size_average=False)
        if size_average:
            loss /= float(target.numel())
        return loss

class Dice_bpm(nn.Module):
    """
    N-D dice for segmentation
    """
    def __init__(self):
        super(Dice_bpm, self).__init__()

    def forward(self, y_pred, y_true, bpm):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred * bpm).sum(dim=vol_axes)
        bottom = torch.clamp(((y_true + y_pred)* bpm).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice

class mask_dice_bpm_inverse(nn.Module):
    def __init__(self):
        super(mask_dice_bpm_inverse, self).__init__()
        self.dice_loss = Dice_bpm()
    def forward(self, pred, target):
        batch_num = pred.size()[0]
        labels = pred.size()[1]
        inds = torch.arange(0, batch_num, dtype=torch.long, device=pred.device)
        focal_weight = torch.zeros((pred.size()), dtype=torch.float32, device=pred.device)
        pred_sm = F.softmax(pred, dim=1).float()
        for label in range(0, labels):
            pred_slice = pred_sm[inds, label].unsqueeze(1)          
            w = 1
            rp_padding = torch.nn.ReflectionPad3d(w)
            p = rp_padding(pred_slice)
            p = torch.clamp(p, min=0.071)

            # laplacian_kernel_1 3D 中间值为 27 
            laplacian_kernel_1 = torch.zeros((2*w+1, 2*w+1, 2*w+1), dtype=torch.float32, device=pred.device).reshape(1,1,2*w+1,2*w+1,2*w+1).requires_grad_(False) - 1   
            laplacian_kernel_1[0,0,w,w,w] = (2*w+1)*(2*w+1)*(2*w+1) - 1
            
            # laplacian_kernel_2 = np.array([[[0,0,0], [0,-1,0], [0,0,0]],\
            #                            [[0,-1,0],[-1,7,-1],[0,-1,0]],\
            #                            [[0,0,0], [0,-1,0], [0,0,0]]])
            # laplacian_kernel_2 = torch.as_tensor(laplacian_kernel_2, dtype=torch.float32, device=pred.device).reshape(1,1,2*w+1,2*w+1,2*w+1)
        
            boundary_target = F.conv3d(p, laplacian_kernel_1)
            boundary_target = torch.abs(boundary_target)
            # save_nii(boundary_target, '/temp4/Reg-Seg/Figure/boundary/_boundary_max-_{}.nii'.format(label))
            # boundary_target = boundary_target.squeeze(1)
            # boundary_target = boundary_target / torch.mean(torch.mean(torch.mean(boundary_target, -1), -1), -1)
            focal_weight[inds, label,:] = boundary_target.detach()
            # save_nii(boundary_target, '/temp4/Reg-Seg/Figure/boundary/boundary_{}.nii'.format(label))

        # focal_weight_avg = torch.mean(focal_weight, 1).unsqueeze(1)
        # save_nii(focal_weight_avg, '/temp4/Reg-Seg/Figure/boundary/1/seg/30_boundary_avg.nii')
        focal_weight = 0.5 / (focal_weight + 0.5)
        # save_nii(focal_weight_avg, '/temp4/Reg-Seg/Figure/boundary/1/seg/30_boundary_reverse_avg.nii')
        # import pdb; pdb.set_trace()
        dice_loss = self.dice_loss(pred, target, focal_weight)
        # import pdb; pdb.set_trace()
        return dice_loss

class Dice(nn.Module):
    """
    N-D dice for segmentation
    """
    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice

def dice_coef(y_true, y_pred):
    smooth = 1.
    a = torch.sum(y_true * y_pred, (2, 3, 4))
    b = torch.sum(y_true**2, (2, 3, 4))
    c = torch.sum(y_pred**2, (2, 3, 4))
    dice = (2 * a + smooth) / (b + c + smooth)
    return 1 - torch.mean(dice)

def MSE(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def js_div(p_output, q_output, get_softmax=True): 
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='mean')
    p_log = F.log_softmax(p_output, dim=1)
    q_log = F.log_softmax(q_output, dim=1)
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    # log_mean_output = ((p_output + q_output)/2).log()
    # return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2
    log_mean_output = ((p_output + q_output)/2)
    return (KLDivLoss(p_log, log_mean_output) + KLDivLoss(q_log, log_mean_output))/2
    
def nas_ncc(I, J):
    
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    win = [9] * ndims
    sum_filt = torch.ones([1, 1, *win]).to("cuda")
    pad_no = math.floor(win[0] / 2)
    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1 * torch.mean(cc)
    # return -1 * torch.mean(cc) + 1

def LoG(pred):
    batch_num = pred.size()[0]
    labels = pred.size()[1]
    inds = torch.arange(0, batch_num, dtype=torch.long, device=pred.device)
    focal_weight = torch.zeros((pred.size()), dtype=torch.float32, device=pred.device)
    for label in range(0, labels):

        pred_slice = pred[inds, label].unsqueeze(1)

        w = 1
        rp_padding = torch.nn.ReflectionPad3d(w)
        p = rp_padding(pred_slice)
        p = torch.clamp(p, min=0.071)
        

        # laplacian_kernel_1 3D 中间值为 27 
        laplacian_kernel_1 = torch.zeros((2*w+1, 2*w+1, 2*w+1), dtype=torch.float32, device=pred.device).reshape(1,1,2*w+1,2*w+1,2*w+1).requires_grad_(False) - 1   
        laplacian_kernel_1[0,0,w,w,w] = (2*w+1)*(2*w+1)*(2*w+1) - 1
       
        boundary_target = F.conv3d(p, laplacian_kernel_1)
        boundary_target = torch.abs(boundary_target)

        focal_weight[inds, label,:] = boundary_target.detach()

    focal_weight_avg = torch.sum(focal_weight, 1).unsqueeze(1)

    return focal_weight_avg  

def LoG_ncc(I, J, pred, labels): 
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    win = [9] * ndims
    sum_filt = torch.ones([1, 1, *win]).to("cuda")
    pad_no = math.floor(win[0] / 2)
    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)
        
    pred = F.softmax(pred, dim=1).float()
    weight = LoG(pred)
    # weight_ = LoG_sm(pred)
    # import pdb; pdb.set_trace()
    
    # I_var, J_var, cross = compute_local_sums(weight*I, weight*J, sum_filt, stride, padding, win)
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    
    ncc = weight * cc

    return -1 * torch.mean(ncc)

def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)


    # I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    # J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    # I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    # J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    # IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross

# Hausdorff and ASSD evaluation
def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if(dim == 2):
        strt = ndimage.generate_binary_structure(2,1)
    else:
        strt = ndimage.generate_binary_structure(3,1)
    ero  = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8) 
    return edge 


def binary_hd95(s, g, spacing = None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert(image_dim == len(g.shape))
    if(spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert(image_dim == len(spacing))
    img = np.zeros_like(s)
    if(image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif(image_dim ==3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    a = int(len(dist_list1)*0.95)
    dist1 = dist_list1[int(len(dist_list1)*0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    dist2 = dist_list2[int(len(dist_list2)*0.95)]
    return max(dist1, dist2)


def binary_assd(s, g, spacing = None):
    """
    get the average symetric surface distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert(image_dim == len(g.shape))
    if(spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert(image_dim == len(spacing))
    img = np.zeros_like(s)
    if(image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif(image_dim ==3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng) 
    return assd



class LossFunction_Reg_Seg(nn.Module):
    def __init__(self,):
        super(LossFunction_Reg_Seg, self).__init__()
        self.gradient_loss = gradient_loss()
        self.seg_dice = LossFunction_dice()
        self.spatial_transform_1 = SpatialTransformer() # 针对标签的变形，用双线性插值法
        self.spatial_transform_2 = SpatialTransformer(mode='nearest') # 针对图像的变形，需要用临近算法。

    def forward(self, y, x_y, ylogits, x_seg, flow, n_class):
        ## Reg
        hyper_1 = 10
        hyper_2 = 15
        ncc = nas_ncc(y, x_y)
        grad = self.gradient_loss(flow)
        ## Seg
        dice_seg = self.seg_dice(F.softmax(ylogits, dim=1).float(), x_seg, flow, n_class)
        loss = hyper_1 * ncc + hyper_2 * grad + dice_seg
        return loss, ncc, grad, dice_seg
    

############  loss search
class LossFunction_serach_Reg(nn.Module):
    def __init__(self) -> None:
        super(LossFunction_serach_Reg, self).__init__()
        ## dice for reg
        self.dice = LossFunction_dice()
        self.grad_loss = gradient_loss()
        self.spatial_transform_1 = SpatialTransformer() # 针对标签的变形，用双线性插值法
        self.spatial_transform_2 = SpatialTransformer(mode='nearest') # 针对图像的变形，需要用临近算法。

    def forward(self, x, y, x2y, y2x, x2y_flow, y2x_flow, ylogits, x_seg, hyper):
        print(hyper[0].is_cuda)
        dice_loss = hyper[0]*self.dice(
            F.softmax(ylogits, dim=1).float(), x_seg, x2y_flow, 14)
        ncc_loss = hyper[1]*(nas_ncc(y, x2y)+nas_ncc(x,y2x))
        grad_loss = hyper[2]*(self.grad_loss(x2y_flow)+self.grad_loss(y2x_flow))
        i_loss = hyper[3]*MSE(-1*self.spatial_transform_1(x2y_flow,x2y_flow), y2x_flow)

        loss = dice_loss + ncc_loss + grad_loss + i_loss
        return loss, dice_loss, ncc_loss, grad_loss, i_loss