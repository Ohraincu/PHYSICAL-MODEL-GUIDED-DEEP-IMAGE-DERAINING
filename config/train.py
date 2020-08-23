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
from dataset import TrainValDataset,TestDataset
from model import Net,Discriminator_rain_img,Discriminator_img
# from model import VGG
from cal_ssim import SSIM

logger = settings.logger
os.environ['CUDA_VISIBLE_DEVICES'] = settings.device_id
device_ids=range(torch.cuda.device_count())
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
#torch.cuda.set_device(settings.device_id)
import numpy as np


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
        ensure_dir('../log_test')
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
        self.l2 = MSELoss().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.ssim = SSIM().cuda()
        # self.vgg = VGG().cuda()
        self.bceloss = nn.BCELoss().cuda()
        self.step = 0
        self.ssim_val = 0
        self.psnr_val = 0
        self.save_steps = settings.save_steps
        self.num_workers = settings.num_workers
        self.batch_size = settings.batch_size
        self.writers = {}
        self.dataloaders = {}

        self.opt_net = Adam(self.net.parameters(), lr=settings.lr)
        self.sche_net = MultiStepLR(self.opt_net, milestones=[settings.l1, settings.l2], gamma=0.1)

        self.opt_dis_rain_img = Adam(self.dis_rain_img.parameters(), lr=settings.lr)

        self.sche_dis_rain_img = MultiStepLR(self.opt_dis_rain_img, milestones=[settings.l1, settings.l2], gamma=0.1)

        self.opt_dis_img = Adam(self.dis_img.parameters(), lr=settings.lr)
        self.sche_dis_img = MultiStepLR(self.opt_dis_img, milestones=[settings.l1, settings.l2],gamma=0.1)

    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]

    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(k, v, self.step)

        out['lr'] = self.opt_net.param_groups[0]['lr']
        out['step'] = self.step
        outputs = [
            "{}:{:.4g}".format(k, v) 
            for k, v in out.items()
        ]
        logger.info(name + '--' + ' '.join(outputs))

    def get_dataloader(self, dataset_name):
        dataset = TrainValDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=self.batch_size, 
                            shuffle=True, num_workers=self.num_workers, drop_last=True)
        return iter(self.dataloaders[dataset_name])

    def get_test_dataloader(self, dataset_name):
        dataset = TestDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=1,
                           shuffle=True, num_workers=self.num_workers, drop_last=True)
        return iter(self.dataloaders[dataset_name])

    def save_checkpoints_net(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'net': self.net.state_dict(),
            'clock_net': self.step,
            'opt_net': self.opt_net.state_dict(),
        }
        torch.save(obj, ckp_path)

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

    def save_checkpoints_dis_rain_img(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'dis_rain_img': self.dis_rain_img.state_dict(),
            'clock_dis_rain_img': self.step,
            'opt_dis_rain_img': self.opt_dis_rain_img.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints_dis_rain_img(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            logger.info('Load checkpoint %s' % ckp_path)
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.dis_rain_img.load_state_dict(obj['dis_rain_img'])
        self.opt_dis_rain_img.load_state_dict(obj['opt_dis_rain_img'])
        self.step = obj['clock_dis_rain_img']
        self.sche_dis_rain_img.last_epoch = self.step

    def save_checkpoints_dis_img(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'dis_img': self.dis_img.state_dict(),
            'clock_dis_img': self.step,
            'opt_dis_img': self.opt_dis_img.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints_dis_img(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            logger.info('Load checkpoint %s' % ckp_path)
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.dis_img.load_state_dict(obj['dis_img'])
        self.opt_dis_img.load_state_dict(obj['opt_dis_img'])
        self.step = obj['clock_dis_img']
        self.sche_dis_img.last_epoch = self.step

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()  # 1. torch.numel() 返回一个tensor变量内所有元素个数，可以理解为矩阵内元素的个数
        print(model)
        print("The number of parameters: {}".format(num_params))

    def loss_vgg(self,input,groundtruth):
        vgg_gt = self.vgg.forward(groundtruth)
        eval = self.vgg.forward(input)
        loss_vgg = [self.l1(eval[m], vgg_gt[m]) for m in range(len(vgg_gt))]
        loss = sum(loss_vgg)
        return loss

    def loss_dis(self, output, label):
        label_d = torch.FloatTensor(settings.batch_size)
        label_d = Variable(label_d.cuda())
        if label == 0:
            label_d.resize_((settings.batch_size, 1, settings.sizePatchGAN, settings.sizePatchGAN)).fill_(0)
        elif label == 1:
            label_d.resize_((settings.batch_size, 1, settings.sizePatchGAN, settings.sizePatchGAN)).fill_(1)
        #print(output)
        loss = self.bceloss(output, label_d)
        #print(loss)
        return loss

    def train_dis_rain_img(self, batch):
        O, B = batch['O'].cuda(), batch['B'].cuda()
        R = O - B
        O, B, R = Variable(O, requires_grad=False), Variable(B, requires_grad=False),Variable(R, requires_grad=False)

        for p in self.dis_rain_img.parameters():
            p.requires_grad = True

        self.dis_rain_img.zero_grad()
        img, derain, rain = self.net(O)
        img, derain, rain = img.detach(), derain.detach(), rain.detach()
        loss_fake = self.loss_dis(self.dis_rain_img(torch.cat([derain, rain], dim=1)), 0)
        loss_fake.backward()
        loss_real = self.loss_dis(self.dis_rain_img(torch.cat([B, R], dim=1)), 1)
        loss_real.backward()
        loss = loss_fake + loss_real
        self.opt_dis_rain_img.step()
        losses = {
            'loss_dis_rain_img': loss
        }
        self.write('train', losses)

    def train_dis_rain(self, batch):
        O, B = batch['O'].cuda(), batch['B'].cuda()
        R = O - B
        O, B, R = Variable(O, requires_grad=False), Variable(B, requires_grad=False), Variable(R, requires_grad=False)

        for p in self.dis_img.parameters():
            p.requires_grad = True
        self.dis_img.zero_grad()
        img, derain, rain = self.net(O)
        img, derain, rain = img.detach(), derain.detach(), rain.detach()
        loss_fake = self.loss_dis(self.dis_img(img), 0)
        loss_fake.backward()
        loss_real = self.loss_dis(self.dis_img(B), 1)
        loss_real.backward()
        loss = loss_fake + loss_real

        self.opt_dis_img.step()
        losses = {
            'loss_dis_rain': loss
        }
        self.write('train', losses)

    def inf_batch(self, name, batch):
        if name == 'train':
            self.net.zero_grad()
        if self.step == 0:
            self.print_network(self.net)
        for p in self.dis_rain_img.parameters():
            p.requires_grad = False
        for p in self.dis_img.parameters():
            p.requires_grad = False
        O, B = batch['O'].cuda(), batch['B'].cuda()
        R = O - B
        O, B, R = Variable(O, requires_grad=False), Variable(B, requires_grad=False), Variable(R, requires_grad=False)
        img, derain, rain = self.net(O)
        if settings.network_style == 'only_rain':
            loss_rain = self.l1(rain,R)
            loss = settings.l_rain * loss_rain
            ssim_list = [self.ssim(O - R, B) for R in [R]]
        if settings.network_style == 'only_derain':
            loss_derain = self.l1(derain, B)
            loss = settings.l_derain * loss_derain
            ssim_list = [self.ssim(derain, B) for derain in [derain]]
        if settings.network_style == 'rain_derain_no_guide':
            loss_rain = self.l1(rain, R)
            loss_derain = self.l1(derain, B)
            loss_img = self.l1(img, B)
            loss_cycle = self.l1(derain+rain, O)
            loss = loss_img + settings.l_rain * loss_rain + settings.l_derain * loss_derain + settings.l_rain_derain * loss_cycle
            ssim_list = [self.ssim(img, B) for img in [img]]
        if settings.network_style == 'rain_derain_with_guide':
            loss_rain = self.l1(rain, R)
            loss_derain = self.l1(derain, B)
            loss_cycle = self.l1(derain+rain,O)
            loss_img = self.l1(img, B)
            loss = loss_img + settings.l_rain * loss_rain + settings.l_derain * loss_derain + settings.l_rain_derain * loss_cycle
            ssim_list = [self.ssim(img, B) for img in [img]]
        loss_list_img = [loss]

        if settings.gan is True:
            loss_dis_rain_img = self.loss_dis(self.dis_rain_img(torch.cat([derain, rain], dim=1)), 1)
            loss_dis_img = self.loss_dis(self.dis_img(img), 1)
            loss = loss + settings.l_dis_rain_derain * loss_dis_rain_img + settings.l_dis_img * loss_dis_img
        else:
            loss = loss
        if name == 'train':
            loss.backward()
            self.opt_net.step()

        losses = {
            'loss%d' % i: loss.item()
            for i, loss in enumerate(loss_list_img)
        }
        ssimes = {
            'ssim%d' % i: ssim.item()
            for i, ssim in enumerate(ssim_list)
        }
        losses.update(ssimes)
        self.write(name, losses)

        return img

    def inf_test_batch(self, name, batch):
        O, B = batch['O'].cuda(), batch['B'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        R = O - B
        with torch.no_grad():
            img, derain, rain = self.net(O)
        if settings.network_style == 'only_rain':
            loss_rain = self.l1(rain, R)
            loss = settings.l_rain * loss_rain
            ssim = self.ssim(O - R, B)
            psnr = PSNR((O - R).data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
        if settings.network_style == 'only_derain':
            loss_derain = self.l1(derain,B)
            loss = settings.l_derain * loss_derain
            ssim = self.ssim(derain, B)
            psnr = PSNR(derain.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
        if settings.network_style == 'rain_derain_no_guide':
            loss_rain = self.l1(rain, R)
            loss_derain = self.l1(derain, B)
            loss_img = self.l1(img, B)
            loss_cycle = self.l1(derain+rain, O)
            loss = loss_img + settings.l_rain * loss_rain + settings.l_derain * loss_derain + settings.l_rain_derain * loss_cycle
            ssim = self.ssim(img, B)
            psnr = PSNR(img.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
        if settings.network_style == 'rain_derain_with_guide':
            loss_rain = self.l1(rain, R)
            loss_derain = self.l1(derain, B)
            loss_cycle = self.l1(derain+rain,O)
            loss_img = self.l1(img, B)
            loss = loss_img + settings.l_rain * loss_rain + settings.l_derain * loss_derain + settings.l_rain_derain * loss_cycle
            ssim = self.ssim(img, B)
            psnr = PSNR(img.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
        loss = loss.data.cpu().numpy()

        return loss, ssim.data.cpu().numpy(), psnr

    def save_image(self, name, img_lists):
        data, pred, label = img_lists
        pred = pred.cpu().data

        data, label, pred = data * 255, label * 255, pred * 255
        pred = np.clip(pred, 0, 255)

        h, w = pred.shape[-2:]

        gen_num = (1, 1)
        img = np.zeros((gen_num[0] * h, gen_num[1] * 3 * w, 3))
        for img_list in img_lists:
            for i in range(gen_num[0]):
                row = i * h
                for j in range(gen_num[1]):
                    idx = i * gen_num[1] + j
                    tmp_list = [data[idx], pred[idx], label[idx]]
                    for k in range(3):
                        col = (j * 3 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
                        img[row: row+h, col: col+w] = tmp 

        img_file = os.path.join(self.log_dir, '%d_%s.jpg' % (self.step, name))
        cv2.imwrite(img_file, img)


def run_train_val(ckp_name_net='latest_net',
                  ckp_name_dis_rain_img='latest_dis_rain_img',
                  ckp_name_dis_img='latest_dis_img',):
    sess = Session()
    # only_rain,
    # only_derain,
    # rain_derain_no_guide,
    # rain_derain_with_guide,
    # rain_derain_with_guide

    sess.load_checkpoints_net(ckp_name_net)
    if settings.gan is True:
        sess.load_checkpoints_dis_rain_img(ckp_name_dis_rain_img)
        sess.load_checkpoints_dis_img(ckp_name_dis_img)

    sess.tensorboard('train')

    dt_train = sess.get_dataloader('train')
    # dt_val = sess.get_dataloader('val')

    while sess.step < settings.total_step:
        sess.sche_net.step()
        if settings.gan is True:
            sess.sche_dis_rain_img.step()
            sess.sche_dis_img.step()
        sess.net.train()
        if settings.gan is True:
            sess.dis_rain_img.train()
            sess.dis_img.train()
        try:
            batch_t = next(dt_train)
        except StopIteration:
            dt_train = sess.get_dataloader('train')
            batch_t = next(dt_train)
        pred_t = sess.inf_batch('train', batch_t)
        if settings.gan is True:
            sess.train_dis_rain(batch_t)
            sess.train_dis_rain_img(batch_t)
        ssim_all = 0
        psnr_all = 0
        loss_all = 0
        num_all = 0
        if sess.step % settings.one_epoch_step == 0:
            dt_val = sess.get_test_dataloader('test')
            sess.net.eval()

            for i, batch_v in enumerate(dt_val):
                print(i)
                loss, ssim, psnr = sess.inf_test_batch('test', batch_v)
                ssim_all = ssim_all + ssim
                psnr_all = psnr_all + psnr
                loss_all = loss_all + loss
                num_all = num_all + 1
            loss_avg = loss_all / num_all
            ssim_avg = ssim_all / num_all
            psnr_avg = psnr_all / num_all
            if ssim_avg > sess.ssim_val:
                sess.ssim_val = ssim_avg
                sess.psnr_val = psnr_avg
                sess.save_checkpoints_net('best_net')
                sess.save_checkpoints_dis_rain_img('best_dis_rain_img_%d')
                sess.save_checkpoints_dis_img('best_dis_img_%d')
                    # Logging
            logfile = open(
                '../log_test/' + 'val' + '.txt',
                'a+'
            )
            epoch = int(sess.step / settings.one_epoch_step)
            logfile.write(
                'step  = ' + str(sess.step) + '\t'
                'epoch = ' + str(epoch) + '\t'
                'loss  = ' + str(loss_avg) + '\t'
                'ssim  = ' + str(ssim_avg) + '\t'
                'pnsr  = ' + str(psnr_avg) + '\t'
                '\n\n'
            )
            logfile.close()

        if sess.step % int(sess.save_steps / 16) == 0:
            sess.save_checkpoints_net('latest_net')
            sess.save_checkpoints_dis_img('latest_dis_img')
            sess.save_checkpoints_dis_rain_img('latest_dis_rain_img')
        if sess.step % int(sess.save_steps / 2) == 0:
            sess.save_image('train', [batch_t['O'], pred_t, batch_t['B']])
            #    if sess.step % 4 == 0:
            #        sess.save_image('val', [batch_v['O'], pred_v, batch_v['B']])
            #        logger.info('save image as step_%d' % sess.step)
        if sess.step % sess.save_steps == 0:
            sess.save_checkpoints_net('step_net_%d' % sess.step)
            sess.save_checkpoints_dis_rain_img('step_dis_rain_img_%d' % sess.step)
            sess.save_checkpoints_dis_img('step_dis_img_%d' % sess.step)
            logger.info('save model as step_net_%d' % sess.step)
        sess.step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--model_1', default='latest_net')
    parser.add_argument('-m2', '--model_2', default='latest_dis_rain_img')
    parser.add_argument('-m3', '--model_3', default='latest_dis_img')
    args = parser.parse_args(sys.argv[1:])
    run_train_val(args.model_1, args.model_2, args.model_3)


