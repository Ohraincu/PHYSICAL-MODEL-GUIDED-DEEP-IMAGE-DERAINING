import os
import sys
import cv2
import argparse
import math
import numpy as np
import itertools

import torch
from torch import nn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import settings
from dataset import ShowDataset
from model import Net, Discriminator_rain_img,Discriminator_img
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
        self.show_dir = settings.show_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.show_dir)
        ensure_dir(settings.model_dir)
        logger.info('set show dir as %s' % settings.show_dir)
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
        self.ssim = SSIM().cuda()
        self.dataloaders = {}
        self.ssim = SSIM().cuda()

    def get_dataloader(self, dataset_name):
        dataset = ShowDataset(dataset_name)
        self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=1, 
                            shuffle=False, num_workers=1)
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

    def inf_batch(self, name, batch):
        with torch.no_grad():
            O, B = batch['O'].cuda(), batch['B'].cuda()
            R = O - B
            O, B, R = Variable(O, requires_grad=False), Variable(B, requires_grad=False), Variable(R, requires_grad=False)
            img, derain, rain = self.net(O)
            if settings.network_style == 'only_rain':
                img, derain, rain = self.net(O)
                img = O - rain
            if settings.network_style == 'only_derain':
                img, derain, rain = self.net(O)
                img = derain
            if settings.network_style == 'rain_derain_no_guide':
                img, derain, rain = self.net(O)
            if settings.network_style == 'rain_derain_with_guide':
                img, derain, rain = self.net(O)
        ssim = self.ssim(img, B)
        psnr = PSNR(img.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
        print('ssim:%8f--------------------------------psnr:%8f'%(ssim, psnr))

        return img, derain, rain, psnr, ssim

    def save_image(self, No, imgs, name, psnr, ssim):
        for i, img in enumerate(imgs):
            img = (img.cpu().data * 255).numpy()
            img = np.clip(img, 0, 255)
            img = np.transpose(img, (1, 2, 0))
            h, w, c = img.shape

            img_file = os.path.join(self.show_dir, '%s_%d_%d_%4f_%4f.png' % (name, No, i, psnr, ssim))
            cv2.imwrite(img_file, img)


def run_show(ckp_name_net='latest_net'):
    sess = Session()
    sess.net.eval()
    sess.load_checkpoints_net(ckp_name_net)
    dataset = 'test_real'
    dt = sess.get_dataloader(dataset)

    for i, batch in enumerate(dt):
        logger.info(i)
        img, derain, rain, psnr, ssim = sess.inf_batch('test', batch)
        sess.save_image(i, img, dataset, psnr, ssim)
        sess.save_image(i, derain, dataset + 'derain', psnr, ssim)
        sess.save_image(i, rain, dataset + 'rain', psnr, ssim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--model_1', default='latest_net')
    args = parser.parse_args(sys.argv[1:])
    run_show(args.model_1)

