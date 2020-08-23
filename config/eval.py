import os
import sys
import cv2
import argparse
import numpy as np
import math
import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import settings
from dataset import TestDataset
from model import Net, Discriminator_rain_img,Discriminator_img
from model import VGG
from cal_ssim import SSIM

logger = settings.logger
os.environ['CUDA_VISIBLE_DEVICES'] = settings.device_id
device_ids=range(torch.cuda.device_count())
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
#torch.cuda.set_device(settings.device_id)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
def PSNR(img1, img2):
    b,_,_,_=img1.shape
    #mse=0
    #for i in range(b):
    img1=np.clip(img1,0,255)
    img2=np.clip(img2,0,255)
    mse = np.mean((img1/ 255. - img2/ 255.) ** 2)#+mse
    if mse == 0:
        return 100
    #mse=mse/b
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))        
class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)
        if torch.cuda.is_available():
            self.net = Net().cuda()
            self.dis_rain_img = Discriminator_rain_img().cuda()
            self.dis_img = Discriminator_img().cuda()
        if len(device_ids) > 1:
            self.net = nn.DataParallel(Net()).cuda()
            self.dis_rain_img = nn.DataParallel(Discriminator_rain_img()).cuda()
            self.dis_img = nn.DataParallel(Discriminator_img()).cuda()
        self.opt_net = Adam(self.net.parameters(), lr=settings.lr)
        self.sche_net = MultiStepLR(self.opt_net, milestones=[settings.l1, settings.l2], gamma=0.1)

        self.opt_dis_rain_img = Adam(self.dis_rain_img.parameters(), lr=settings.lr)
        self.sche_dis_rain_img = MultiStepLR(self.opt_dis_rain_img, milestones=[settings.l1, settings.l2], gamma=0.1)
        self.opt_dis_img = Adam(self.dis_rain_img.parameters(), lr=settings.lr)
        self.sche_dis_img = MultiStepLR(self.opt_dis_rain_img, milestones=[settings.l1, settings.l2], gamma=0.1)


        self.l2 = MSELoss().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.ssim = SSIM().cuda()
        self.vgg = VGG().cuda()
        self.dataloaders = {}

    def get_dataloader(self, dataset_name):
        dataset = TestDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=1, 
                            shuffle=False, num_workers=1, drop_last=False)
        return self.dataloaders[dataset_name]

    def load_checkpoints_net(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            logger.info('Load checkpoint %s' % ckp_path)
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
        self.opt_net.load_state_dict(obj['opt_net'])
        self.step = obj['clock_net']
        self.sche_net.last_epoch = self.step

    def load_checkpoints_dis_rain_img(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            logger.info('Load checkpoint %s' % ckp_path)
            obj = torch.load(ckp_path, map_location={'cuda:1':'cuda:0','cuda:2':'cuda:0','cuda:3':'cuda:0','cuda:4':'cuda:0','cuda:5':'cuda:0','cuda:6':'cuda:0','cuda:7':'cuda:0'})
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.dis_rain_img.load_state_dict(obj['dis_rain_img'])
        self.opt_dis_rain_img.load_state_dict(obj['opt_dis_rain_img'])
        self.step = obj['clock_dis_rain_img']
        self.sche_dis_rain_img.last_epoch = self.step

    def load_checkpoints_dis_img(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            logger.info('Load checkpoint %s' % ckp_path)
            obj = torch.load(ckp_path, map_location={'cuda:1':'cuda:0','cuda:2':'cuda:0','cuda:3':'cuda:0','cuda:4':'cuda:0','cuda:5':'cuda:0','cuda:6':'cuda:0','cuda:7':'cuda:0'})
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.dis_img.load_state_dict(obj['dis_img'])
        self.opt_dis_img.load_state_dict(obj['opt_dis_img'])
        self.step = obj['clock_dis_img']
        self.sche_dis_img.last_epoch = self.step

    def loss_vgg(self,input,groundtruth):
        vgg_gt = self.vgg.forward(groundtruth)
        eval = self.vgg.forward(input)
        loss_vgg = [self.l1(eval[m], vgg_gt[m]) for m in range(len(vgg_gt))]
        loss = sum(loss_vgg)
        return loss

    def inf_batch(self, name, batch):
        with torch.no_grad():
            O, B = batch['O'].cuda(), batch['B'].cuda()
            R = O - B
            O, B, R = Variable(O, requires_grad=False), Variable(B, requires_grad=False), Variable(R, requires_grad=False)
            img, derain, rain = self.net(O)
            if settings.network_style == 'only_rain':
                img, derain, rain = self.net(O)
                img = O-rain
            if settings.network_style == 'only_derain':
                img, derain, rain = self.net(O)
                img = derain
            if settings.network_style == 'rain_derain_no_guide':
                img, derain, rain = self.net(O)
            if settings.network_style == 'rain_derain_with_guide':
                img, derain, rain = self.net(O)
        loss_list_img = [self.l1(img, B) for img in [img]]
        ssim_list = [self.ssim(img, B) for img in [img]]
        psnr = PSNR(img.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
        losses = {
            'loss%d' % i: loss.item()
            for i, loss in enumerate(loss_list_img)
        }
        ssimes = {
            'ssim%d' % i: ssim.item()
            for i, ssim in enumerate(ssim_list)
        }
        losses.update(ssimes)

        return losses, psnr


def run_test(ckp_name_net='latest_net'):
    sess = Session()
    sess.net.eval()
    sess.load_checkpoints_net(ckp_name_net)
    dt = sess.get_dataloader('test')
    psnr_all = 0
    all_num = 0
    all_losses = {}
    for i, batch in enumerate(dt):
        losses, psnr = sess.inf_batch('test', batch)
        psnr_all = psnr_all + psnr
        batch_size = batch['O'].size(0)
        all_num += batch_size
        for key, val in losses.items():
            if i == 0:
                all_losses[key] = 0.
            all_losses[key] += val * batch_size
            logger.info('batch %d mse %s: %f' % (i, key, val))

    for key, val in all_losses.items():
        logger.info('total mse %s: %f' % (key, val / all_num))
    # psnr=sum(psnr_all)
    # print(psnr)
    print('psnr_ll:%8f' % (psnr_all / all_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--model_1', default='latest_net')
    args = parser.parse_args(sys.argv[1:])
    run_test(args.model_1)

