# 使用我们本身的网络框架训练iseg
import sys
from ast import arg
import string
from torch.utils.tensorboard import SummaryWriter
import os, glob, losses3D
import argparse
from torch.utils.data import DataLoader
from data import datasets_mixed as datasets
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models2 import *
import utils.utils as utils
import torch.nn.functional as F
import utils.augmentation as aug
import random
from datetime import datetime
import copy

shape = (128, 128, 128)
torch.set_printoptions(precision=8)

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"/logfile.log", "a")
        print('save path: ', save_dir)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure messages are written to the file immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush() # Ensure the log file is flushed properly

    def close(self):
        self.log.close()  # Close the log file properly

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def main():
    parser = argparse.ArgumentParser()
    # 每次训练前需检查的路径是否正确
    parser.add_argument('--cover_path', type=bool, default=False,help="正式训练时改为ture,调试阶段为false")
    parser.add_argument('--save_path', type=str, default='/root/data1/wangjia/1RegSeg_sam/results/base',help="all log save path")
    parser.add_argument('--train_path', type=str, default='/root/data1/wangjia/DATA/OASIS/TrainVal/*',help="train dataset path")
    parser.add_argument('--test_path', type=str, default='/root/data1/wangjia/DATA/OASIS/Testvol/*',help="test dataset path")
    parser.add_argument('--src_path', type=str, default='/root/data1/wangjia/Unet-RegSeg/Data/atlas/atlas_vol.nii',help="atlas image path")
    parser.add_argument('--sseg_path', type=str, default='/root/data1/wangjia/Unet-RegSeg/Data/atlas/atlas_seg_14labels.nii',help="atlas label path")
    parser.add_argument('--pre_path', type=str, default='/root/data1/wangjia/1RegSeg_sam/checkpoints/best_seg_156.ckpt',help="pretrain model path")

    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--epochs', type=int, default=2000, help="the number of global epochs: G")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    # parser.add_argument('--num_users', type=int, default=2, help="number of users: K")
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    batch_size = 1
    best_sdice = 0
    best_rdice = 0
    best_spath = None
    best_rpath = None

    model_dir = args.save_path
    if args.cover_path:
        if os.path.exists(args.save_path):
            new_name = args.save_path + '_archived_' + get_timestamp()
            print('Path already exists. Rename it to [{:s}]'.format(new_name))
            os.rename(args.save_path, new_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    logger = Logger(args.save_path)
    sys.stdout = logger
    sys.stderr = logger
    lr = 0.0001 # learning rate
    epoch_start = 0
    max_epoch = 200 #max traning epoch
    cont_training = False #if continue training
    img_size = shape
    n_class = 14

    '''
    Initialize model
    '''
    model = Architecture(inshape=img_size, n_channels=1, n_classes=n_class).cuda()
    model.train()
    # copy weights
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)
    '''
    Initialize Feature_Extractor 
    '''
    # load存在的模型参数（权重文件），后缀名可能不同    
    pretrained_dict = torch.load(args.pre_path) 
    model_dict = model.state_dict()
    # 关键在于下面这句，从model_dict中读取key、value时，用if筛选掉不需要的网络层 
    pretrained_dict_updated = {key: value for key, value in pretrained_dict.items() if (key in model_dict and 'fea_extractor' in key)}
    model_dict.update(pretrained_dict_updated)
    model.load_state_dict(model_dict)
    # for p in model.fea_extractor.parameters():
    #     p.requires_grad = False
    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model('nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model('bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    # if cont_training:
    #     epoch_start = 25
    #     model_path = '/root/data1/wangxiaolin/Unet-RegSeg/experiment_for_fixed_encoder/alternative-v1/model/24.ckpt'
    #     best_model = torch.load(model_path)
    #     print('Model: {} loaded!'.format(model_path))
    #     model.load_state_dict(best_model)

    '''
    Initialize training
    '''
    spatial_aug = aug.SpatialTransform(do_rotation=True,
                angle_x=(-np.pi / 18, np.pi / 18),
                angle_y=(-np.pi / 18, np.pi / 18),
                angle_z=(-np.pi / 18, np.pi / 18),
                do_scale=True,
                scale=(0.9, 1.1))
    # 为配准解码器和分割解码器设置Adam优化器
    r_optimizer = optim.Adam(model.Reg_Decoder.parameters(), lr=lr)
    s_optimizer = optim.Adam(model.Seg_Decoder.parameters(), lr=lr)
    # 传入两个数据集 
    # path = ['/temp4/wangjia/DATA//ABIDE/train/*', '/temp4/wangjia/DATA/ADNI/train/*', '/temp4/wangjia/DATA/PPMI/train/*']
    # path = '/temp4/wangjia/DATA/HCP/train/*'
    path = "/root/data1/wangjia/DATA/OASIS/TrainVal/*"
    # train_path_table = []
    # for i in range(len(path)):
    #     a = glob.glob(path[i])
    #     train_path_table = train_path_table + a
    
    #     print(len(a))
    # print(len(train_path_table))
    # zz
    train_path_table = glob.glob(path) 
    train_set = datasets.TrainOneShot(args)
    # val_path_table = glob.glob('/root/data1/wangjia/DATA/OASIS/Testvol/*')
    val_set = datasets.InferDataset(args)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) 
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    # print(len(val_loader))
    # zz
    # r_criterion = losses3D.LossFunctionNccIntensity_Registration().cuda()
    # GT在前
    reg_criterion = losses3D.LossFunction_Reg_Seg().cuda()
    seg_criterion = losses3D.LossFunction_dice().cuda()
    writer = SummaryWriter(log_dir=model_dir)
    # print(len(w_locals), len(train_loader), len(train_loader[0]), len(train_loader[1]))
    # zzz

    for epoch in range(epoch_start, args.epochs):
        dice_list0, dice_list1, dice_list2, ncc_list0, ncc_list1, ncc_list2 = [], [], [], [], [], []
        print('Training Starts')
        '''
        Training
        '''
        rreg_all, rgrad_all, rcycle_all = [], [], []
        rdice_RegSeg_all = []
        rtotalloss_all = []

        sdice_RegSeg_all = []

        batch_loss = []
        for iter, (data) in enumerate(train_loader):
            # images, labels = images.to(self.args.device), labels.to(self.args.device)
            data = [torch.tensor(t, dtype=torch.float32).cuda() for t in data]
            x = data[0].cuda()   #x是atlas
            y = data[1].cuda()
            x_seg = data[2].cuda()
            code_spa = spatial_aug.rand_coords(x.shape[2:])
            x = spatial_aug.augment_spatial(x, code_spa)
            x_seg = spatial_aug.augment_spatial(x_seg, code_spa, mode='nearest')
            y = spatial_aug.augment_spatial(y, code_spa)

            #特征提取与分割解码 评估模式 配准解码器 训练模式
            model.fea_extractor.eval()
            model.Seg_Decoder.eval()
            model.Reg_Decoder.train()
            # print(x.shape, y.shape)
            # 使用特征提取器提取特征
            encoded_x = model.fea_extractor(x)
            encoded_y = model.fea_extractor(y)
            # 计算配准后的图像和流场
            x_y, flow = model.Reg_Decoder(x, encoded_x, encoded_y)
            # y_x, flow_y_x = model.Reg_Decoder(y, encoded_y, encoded_x)
            # 计算分割结果的logits
            # xlogits = model.Seg_Decoder(encoded_x)
            y_logits = model.Seg_Decoder(encoded_y)
            loss, ncc, grad, dice_seg = reg_criterion(y, x_y, y_logits, x_seg, flow, n_class)

            # loss, ncc, grad, dice_seg = self.loss_func(y, x2y, y_logits, x_seg, flow)
            r_optimizer.zero_grad()
            loss.backward()
            r_optimizer.step()
            '''
            Training Seg_Decoder
            '''
            model.Reg_Decoder.eval()
            model.Seg_Decoder.train()
            warped_x, flow, xlogits, ylogits = model(x, y)
    
            # Reg Seg loss
            dice_RegSeg = seg_criterion(F.softmax(ylogits, dim=1).float(), x_seg, flow, 14)
            s_optimizer.zero_grad()
            dice_RegSeg.backward()
            s_optimizer.step()

            sdice_RegSeg_all.append(dice_RegSeg.item()) 
            if iter % 10 == 0:
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'Train Reg_Decoder {} Iter {} of {} dice_loss {:.8f} ncc {:.8f} grad_loss {:.8f}'
                        .format(epoch, iter, len(train_loader), dice_seg.item(), ncc.item(), grad.item()))
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Train Seg_decoder {} Iter {} of {} dice_seg {:.8f} '
                        .format(epoch, iter, len(train_loader), dice_RegSeg.item()))
                print('total loss {:.8f}'.format(loss.item()))


        # writer.add_scalar('toalloss', np.mean(rdice_RegSeg_all), epoch)


        # # copy weight to net_glob
        # # net_glob.load_state_dict(w_glob)

        # writer.add_scalar('toalloss', np.mean(rdice_RegSeg_all), epoch)

        '''
        Validation
        '''
        if epoch % 1 == 0:
            with torch.no_grad():
                for data in val_loader:
                    model.eval()
                    # data = [torch.tensor(t, dtype=torch.float32).cuda() for t in data]
                    # x是source y是target
                    x = data[0].cuda()   #x是atlas
                    y = data[1].cuda()
                    x_seg = data[2].cuda()
                    y_seg = torch.tensor(data[3], dtype=torch.long)
                    # print(x.shape, y.shape)
                    # zz
                    warped_x, flow, xlogits, ylogits = model(x,y)

                    def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
                    vals0, _ = utils.dice(x_seg.cpu().numpy(), y_seg.cpu().numpy(), nargout=2)
                    dice_list0.append(np.average(vals0))
                    vals1, _ = utils.dice(def_out.cpu().numpy(), y_seg.cpu().numpy(), nargout=2)
                    dice_list1.append(np.average(vals1))
                    # print(np.average(vals0), np.average(vals1))
                    # print(type(y),type(x), y.shape, x.shape)
                    # zzz
                    ncc0 = -losses3D.nas_ncc(y, x)
                    # warped
                    ncc1 = -losses3D.nas_ncc(y, warped_x)

                    ncc_list0.append(ncc0.item())
                    ncc_list1.append(ncc1.item())

                    full_mask = F.softmax(ylogits, dim=1).float()
                    mask = full_mask.argmax(dim=1)
                    vals2, _ = utils.dice(mask.cpu().numpy().squeeze(), y_seg.cpu().numpy().squeeze(), nargout=2)
                    dice_list2.append(np.average(vals2))
                    # print(np.mean(ncc0.item()), np.mean(ncc1.item()))

            result_dir = os.path.join(args.save_path)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            f_test = open(result_dir + '/result.txt', 'a')

            print('Epoch', epoch)
            f_test.write('Epoch '+str(epoch)+'\n')

            print('------------------------Dice------------------------')
            print(' initial :', np.mean(dice_list0), np.std(dice_list0))
            print('   Reg   :', np.mean(dice_list1), np.std(dice_list1))
            print('   Seg   :', np.mean(dice_list2), np.std(dice_list2))
            f_test.write('------------------------Dice------------------------' + '\n')
            f_test.write(' initial :' + str(np.mean(dice_list0)) + str(np.std(dice_list0)) + '\n')
            f_test.write('   Reg   :' + str(np.mean(dice_list1)) + str(np.std(dice_list1)) +'\n')
            f_test.write('   Seg   :' + str(np.mean(dice_list2)) + str(np.std(dice_list2)) +'\n')

            print('------------------------NCC------------------------')
            print(' initial :', np.mean(ncc_list0), np.std(ncc_list0))
            print('  warped :', np.mean(ncc_list1), np.std(ncc_list1))
            f_test.write('------------------------NCC------------------------' + '\n')
            f_test.write('mean:' + str(np.mean(ncc_list0)) + '\n')
            f_test.write('std:' + str(np.std(ncc_list0)) + '\n')
            f_test.write('warped-mean:' + str(np.mean(ncc_list1)) + '\n')
            f_test.write('warped-std:' + str(np.std(ncc_list1)) + '\n')

            writer.add_scalar('Test_Reg_Dice', np.mean(dice_list1), epoch)
            writer.add_scalar('Test_Seg_Dice', np.mean(dice_list2), epoch)
            writer.add_scalar('Test_Ncc', np.mean(ncc_list1), epoch)
            r_dice = np.mean(dice_list1)
            s_dice = np.mean(dice_list2)

            # # 保存所有模型
            # r_path = os.path.join(args.save_path, 'reg_%d.ckpt' % epoch)
            # s_path = os.path.join(args.save_path, 'seg_%d.ckpt' % epoch)
            # torch.save(model.state_dict(), r_path)
            # torch.save(model.state_dict(), s_path)


            if r_dice>best_rdice:
                best_rdice = r_dice
                if best_rpath is not None:
                    os.remove(best_rpath)
                best_rpath = os.path.join(args.save_path, 'best_reg_%d.ckpt' % epoch)
                torch.save(model.state_dict(), best_rpath)
            if s_dice>best_sdice:
                best_sdice = s_dice
                if best_spath is not None:
                    os.remove(best_spath)
                best_spath = os.path.join(args.save_path, 'best_seg_%d.ckpt' % epoch)
                torch.save(model.state_dict(), best_spath)
            f_test.write('Best Reg dice: {}, Best Reg path: {}'.format(best_rdice, best_rpath))
            f_test.write('Best Seg dice: {}, Best Seg path: {}'.format(best_sdice, best_spath))
    sys.stdout.close() 
    writer.close()

# def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()