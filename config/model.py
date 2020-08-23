import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg19
import settings
import numpy
from itertools import combinations, product

import math

class Res_block(nn.Module):
    def __init__(self):
        super(Res_block, self).__init__()
        self.channel = settings.channel_derain
        self.res = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2),)

    def forward(self, x):
        out = x + self.res(x)
        return out

class MSRB(nn.Module):
    def __init__(self):
        super(MSRB, self).__init__()
        self.num_scale = settings.scale_num
        self.channel = settings.channel_derain
        self.scale = nn.ModuleList()
        if settings.pyramid is True:
            for i in range(self.num_scale - 1):
                self.scale.append(nn.MaxPool2d(2 ** (i + 1), 2 ** (i + 1)))
                self.cat = nn.Sequential(nn.Conv2d(self.num_scale * self.channel, self.channel, 1, 1),
                                         nn.LeakyReLU(0.2))
        self.res = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2),)

    def forward(self, x):
        ori = x
        if settings.pyramid is True:
            b, c, h, w = x.size()
            scale = []
            up = []
            up.append(x)
            for i in range(self.num_scale-1):
                scale.append(self.scale[i](x))
                up.append(F.upsample_bilinear(scale[-1], size=[h, w]))
            x = self.cat(torch.cat(up, dim=1))
        res = self.res(x)
        out = res + ori
        return out


class MSB(nn.Module):
    def __init__(self):
        super(MSB, self).__init__()
        self.num_scale = settings.scale_num
        self.channel = settings.channel_derain
        self.scale = nn.ModuleList()
        for i in range(self.num_scale-1):
            self.scale.append(nn.MaxPool2d(2**(i+1), 2**(i+1)))
        self.dense = Dense_block(self.channel, 3)
        self.cat = nn.Sequential(nn.Conv2d(self.num_scale*self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))

    def forward(self, x):
        if settings.pyramid is True:
            b, c, h, w = x.size()
            scale = []
            up = []
            up.append(x)
            for i in range(self.num_scale-1):
                scale.append(self.scale[i](x))
                up.append(F.upsample_bilinear(scale[-1], size=[h, w]))
            cat = self.cat(torch.cat(up, dim=1))
            dense = self.dense(cat)
            out = x + dense
            return out
        else:
            dense = self.dense(x)
            return x + dense


My_unit = MSRB



class Derain_net_encoder_decoder(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(Derain_net_encoder_decoder, self).__init__()
        self.convert = nn.Sequential(nn.Conv2d(in_channel, mid_channel, 3, 1, 1), nn.LeakyReLU(0.2),)
        self.conv_1 = nn.Sequential(My_unit(),
                                    My_unit()
                                    )

        self.pooling_1 = nn.MaxPool2d(2, 2)
        self.conv_2 = nn.Sequential(My_unit(),
                                    My_unit()
                                    )
        self.pooling_2 = nn.MaxPool2d(2, 2)

        self.conv_3 = nn.Sequential(My_unit(),My_unit())

        self.conv_4 = nn.Sequential(My_unit(),
                                    My_unit()
                                    )
        self.conv_5 = nn.Sequential(My_unit(),
                                    My_unit()
                                    )

        self.out = nn.Sequential(nn.Conv2d(mid_channel, mid_channel, 3, 1, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(mid_channel, out_channel, 1, 1))
        self.cat_1 = nn.Sequential(nn.Conv2d(2*mid_channel, mid_channel, 1, 1), nn.LeakyReLU(0.2))
        self.cat_2 = nn.Sequential(nn.Conv2d(2*mid_channel, mid_channel, 1, 1), nn.LeakyReLU(0.2))
        self.cat2 = nn.Sequential(nn.Conv2d(2*mid_channel, mid_channel, 1, 1), nn.LeakyReLU(0.2))
        self.cat3 = nn.Sequential(nn.Conv2d(2 * mid_channel, mid_channel, 1, 1), nn.LeakyReLU(0.2))
        self.cat4 = nn.Sequential(nn.Conv2d(2 * mid_channel, mid_channel, 1, 1), nn.LeakyReLU(0.2))
        self.cat5 = nn.Sequential(nn.Conv2d(2 * mid_channel, mid_channel, 1, 1), nn.LeakyReLU(0.2))

    def forward(self, x, guide_conv2=None, guide_conv3=None, guide_conv4=None, guide_conv5=None):
        if guide_conv2 is not None:
            convert = self.convert(x)
            conv1 = self.conv_1(convert)
            b1, c1, h1, w1 = conv1.size()
            pooling1 = self.pooling_1(conv1)
            conv2 = self.conv_2(self.cat2(torch.cat([pooling1, guide_conv2], dim = 1)))
            b2, c2, h2, w2 = conv2.size()
            pooling2 = self.pooling_2(conv2)
            conv3 = self.conv_3(self.cat3(torch.cat([pooling2,guide_conv3],dim = 1)))

            conv4 = self.conv_4(self.cat4(torch.cat([self.cat_1(torch.cat([F.upsample_bilinear(conv3, size=[h2, w2]), conv2], dim=1)),guide_conv4],dim = 1)))
            conv5 = self.conv_5(self.cat5(torch.cat([self.cat_2(torch.cat([F.upsample_bilinear(conv4, size=[h1, w1]), conv1], dim=1)),guide_conv5],dim = 1)))
            derain = self.out(conv5)

            return derain
        else:
            convert = self.convert(x)
            conv1 = self.conv_1(convert)
            b1, c1, h1, w1 = conv1.size()
            pooling1 = self.pooling_1(conv1)
            conv2 = self.conv_2(pooling1)
            b2, c2, h2, w2 = conv2.size()
            pooling2 = self.pooling_2(conv2)
            conv3 = self.conv_3(pooling2)

            conv4 = self.conv_4(self.cat_1(torch.cat([F.upsample_bilinear(conv3, size=[h2, w2]), conv2], dim=1)))
            conv5 = self.conv_5(self.cat_2(torch.cat([F.upsample_bilinear(conv4, size=[h1, w1]), conv1], dim=1)))
            derain = self.out(conv5)

            return derain


class Rain_encoder_decoder(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(Rain_encoder_decoder, self).__init__()
        self.convert = nn.Sequential(nn.Conv2d(in_channel, mid_channel, 3, 1, 1), nn.LeakyReLU(0.2), )
        self.conv_1 = nn.Sequential(My_unit(),
                                    My_unit()
                                    )

        self.pooling_1 = nn.MaxPool2d(2, 2)
        self.conv_2 = nn.Sequential(My_unit(),
                                    My_unit()
                                    )
        self.pooling_2 = nn.MaxPool2d(2, 2)

        self.conv_3 = nn.Sequential(My_unit(), My_unit())

        self.conv_4 = nn.Sequential(My_unit(),
                                    My_unit()
                                    )
        self.conv_5 = nn.Sequential(My_unit(),
                                    My_unit()
                                    )

        self.out = nn.Sequential(nn.Conv2d(mid_channel, mid_channel, 3, 1, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(mid_channel, out_channel, 1, 1))
        self.cat_1 = nn.Sequential(nn.Conv2d(2 * mid_channel, mid_channel, 1, 1), nn.LeakyReLU(0.2))
        self.cat_2 = nn.Sequential(nn.Conv2d(2 * mid_channel, mid_channel, 1, 1), nn.LeakyReLU(0.2))

    def forward(self, x):
        convert = self.convert(x)
        conv1 = self.conv_1(convert)
        b1, c1, h1, w1 = conv1.size()
        pooling1 = self.pooling_1(conv1)
        conv2 = self.conv_2(pooling1)
        b2, c2, h2, w2 = conv2.size()
        pooling2 = self.pooling_2(conv2)
        conv3 = self.conv_3(pooling2)

        conv4 = self.conv_4(self.cat_1(torch.cat([F.upsample_bilinear(conv3, size=[h2, w2]), conv2], dim=1)))
        conv5 = self.conv_5(self.cat_2(torch.cat([F.upsample_bilinear(conv4, size=[h1, w1]), conv1], dim=1)))
        rain_streak = self.out(conv5)

        return rain_streak,conv2,conv3,conv4,conv5


class MSDC(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(MSDC, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, mid_channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channel, mid_channel, 5, 1, 2), nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channel, mid_channel, 7, 1, 3), nn.LeakyReLU(0.2))
        self.cat = nn.Sequential(nn.Conv2d(3*mid_channel, mid_channel, 1, 1), nn.LeakyReLU(0.2))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)
        out = self.cat(torch.cat([conv1, conv3, conv5],dim=1))
        return out


class Rain_derain_cat_encoder_decoder(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(Rain_derain_cat_encoder_decoder, self).__init__()
        if settings.msdc is True:
            self.convert = MSDC(in_channel, mid_channel, out_channel)
        else:
            self.convert = nn.Sequential(nn.Conv2d(in_channel, mid_channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv_1 = nn.Sequential(My_unit(),
                                    My_unit()
                                    )

        self.pooling_1 = nn.MaxPool2d(2, 2)
        self.conv_2 = nn.Sequential(My_unit(),
                                    My_unit()
                                    )
        self.pooling_2 = nn.MaxPool2d(2, 2)

        self.conv_3 = nn.Sequential(My_unit(),My_unit())

        self.conv_4 = nn.Sequential(My_unit(),
                                    My_unit()
                                    )
        self.conv_5 = nn.Sequential(My_unit(),
                                    My_unit()
                                    )

        self.out = nn.Sequential(nn.Conv2d(mid_channel, mid_channel, 3, 1, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(mid_channel, out_channel, 1, 1))
        self.cat_1 = nn.Sequential(nn.Conv2d(2 * mid_channel, mid_channel, 1, 1), nn.LeakyReLU(0.2))
        self.cat_2 = nn.Sequential(nn.Conv2d(2 * mid_channel, mid_channel, 1, 1), nn.LeakyReLU(0.2))

    def forward(self, x):
        convert = self.convert(x)
        conv1 = self.conv_1(convert)
        b1, c1, h1, w1 = conv1.size()
        pooling1 = self.pooling_1(conv1)
        conv2 = self.conv_2(pooling1)
        b2, c2, h2, w2 = conv2.size()
        pooling2 = self.pooling_2(conv2)
        conv3 = self.conv_3(pooling2)

        conv4 = self.conv_4(self.cat_1(torch.cat([F.upsample_bilinear(conv3, size=[h2, w2]), conv2], dim=1)))
        conv5 = self.conv_5(self.cat_2(torch.cat([F.upsample_bilinear(conv4, size=[h1, w1]), conv1], dim=1)))
        rain_streak = self.out(conv5)

        return rain_streak

class Rain_derain_no_guide(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(Rain_derain_no_guide, self).__init__()
        self.convert = nn.Sequential(nn.Conv2d(in_channel, mid_channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv_1 = nn.Sequential(My_unit(),
                                    My_unit()
                                    )

        self.pooling_1 = nn.MaxPool2d(2, 2)
        self.conv_2 = nn.Sequential(My_unit(),
                                    My_unit()
                                    )
        self.pooling_2 = nn.MaxPool2d(2, 2)

        self.conv_3 = nn.Sequential(My_unit(),My_unit())

        self.conv_4 = nn.Sequential(My_unit(),
                                    My_unit()
                                    )
        self.conv_5 = nn.Sequential(My_unit(),
                                    My_unit()
                                    )

        self.out = nn.Sequential(nn.Conv2d(mid_channel, mid_channel, 3, 1, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(mid_channel, out_channel, 1, 1))
        self.cat_1 = nn.Sequential(nn.Conv2d(2 * mid_channel, mid_channel, 1, 1), nn.LeakyReLU(0.2))
        self.cat_2 = nn.Sequential(nn.Conv2d(2 * mid_channel, mid_channel, 1, 1), nn.LeakyReLU(0.2))

    def forward(self, x):
        convert = self.convert(x)
        conv1 = self.conv_1(convert)
        b1, c1, h1, w1 = conv1.size()
        pooling1 = self.pooling_1(conv1)
        conv2 = self.conv_2(pooling1)
        b2, c2, h2, w2 = conv2.size()
        pooling2 = self.pooling_2(conv2)
        conv3 = self.conv_3(pooling2)

        conv4 = self.conv_4(self.cat_1(torch.cat([F.upsample_bilinear(conv3, size=[h2, w2]), conv2], dim=1)))
        conv5 = self.conv_5(self.cat_2(torch.cat([F.upsample_bilinear(conv4, size=[h1, w1]), conv1], dim=1)))
        rain_streak = self.out(conv5)

        return rain_streak


class Res(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Res, self).__init__()
        self.in_dim = in_ch
        self.out_dim = out_ch
        self.res = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1), nn.LeakyReLU(0.2),
            )
        if in_ch != out_ch:
            self.convert = nn.Conv2d(self.in_dim,self.out_dim,1,1)

    def forward(self, x):
        if self.in_dim != self.out_dim:
            y = self.convert(x)
        else:
            y = x
        res = self.res(x)
        out = res+y
        return out


class Refinement(nn.Module):
    def __init__(self):
        super(Refinement, self).__init__()
        self.channel = 32
        self.refinement = nn.Sequential(
            Res(3, self.channel),
            Res(self.channel, 2*self.channel),
            Res(2*self.channel, 4*self.channel),
            Res(4*self.channel, 4*self.channel),
            Res(4*self.channel, 2*self.channel),
            Res(2*self.channel, self.channel),
            Res(self.channel, 3),

        )

    def forward(self, x):
        out = self.refinement(x)
        return out


class Derain_net(nn.Module):
    def __init__(self):
        super(Derain_net, self).__init__()
        self.channel = settings.channel_derain
        self.net = Derain_net_encoder_decoder(3, self.channel, 3)

    def forward(self, x, conv2=None, conv3=None, conv4=None, conv5=None):
        derain = self.net(x, conv2, conv3, conv4, conv5)
        return derain


class Rain_streak_net(nn.Module):
    def __init__(self):
        super(Rain_streak_net, self).__init__()
        self.channel = settings.channel_derain
        self.net = Rain_encoder_decoder(3, self.channel, 3)

    def forward(self, x):
        rain,conv2,conv3,conv4,conv5 = self.net(x)
        return rain, conv2, conv3, conv4, conv5

class Rain_img_net(nn.Module):
    def __init__(self):
        super(Rain_img_net, self).__init__()
        self.channel = settings.channel_derain
        self.net = Rain_derain_cat_encoder_decoder(6, self.channel, 3)
    def forward(self, x):
        img = self.net(x)
        return img


class Rain_img_no_guide(nn.Module):
    def __init__(self):
        super(Rain_img_no_guide, self).__init__()
        self.channel = settings.channel_derain
        self.net = Rain_derain_no_guide(3, self.channel, 3)

    def forward(self, x):
        img = self.net(x)
        return img

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if settings.network_style == 'only_rain':
            self.rain = Rain_streak_net()
        if settings.network_style == 'only_derain':
            self.derain = Derain_net()
        if settings.network_style == 'rain_derain_no_guide':
            self.rain = Rain_streak_net()
            self.derain = Derain_net()
            self.img_no_guide = Rain_img_no_guide()
        if settings.network_style == 'rain_derain_with_guide':
            self.rain = Rain_streak_net()
            self.derain = Derain_net()
            self.img_guide = Rain_img_net()

    def forward(self, x):
        if settings.network_style == 'only_rain':
            rain, conv2, conv3, conv4, conv5 = self.rain(x)
            return x-rain, x-rain, rain
        if settings.network_style == 'only_derain':
            derain = self.derain(x)
            return derain, derain, derain
        if settings.network_style == 'rain_derain_no_guide':
            rain, conv2, conv3, conv4, conv5 = self.rain(x)
            derain = self.derain(x)
            img = self.img_no_guide(derain)
            return img, derain, rain
        if settings.network_style == 'rain_derain_with_guide':
            rain, conv2, conv3, conv4, conv5 = self.rain(x)
            derain = self.derain(x)
            img = self.img_guide(torch.cat([derain, rain],dim=1))
            return img, derain, rain

# only_rain,
# only_derain,
# rain_derain_no_guide_no_dis,
# rain_derain_with_guide_no_dis,
# rain_derain_with_guide_and_dis
class Discriminator_rain_img(nn.Module):
    def __init__(self):
        super(Discriminator_rain_img, self).__init__()
        self.res1 = nn.Sequential(nn.Conv2d(6, 64, 3, 1, 1), nn.BatchNorm2d(64),nn.LeakyReLU(0.2))
        self.res2= nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.res3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.res4 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.res5 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.avg_pooling = nn.Sequential(nn.Conv2d(128, 1, 3, 1, 1))
        self.act = nn.Sigmoid()
    def forward(self, x):
        res1 = self.res1(x)
        res2 = self.res2(F.max_pool2d(res1, [2, 2], [2, 2]))
        res3 = self.res3(F.max_pool2d(res2, [2, 2], [2, 2]))
        res4 = self.res4(F.max_pool2d(res3, [2, 2], [2, 2]))
        res5 = self.res5(F.max_pool2d(res4, [2, 2], [2, 2]))
        avg = self.act(self.avg_pooling(F.max_pool2d(res5, [2, 2], [2, 2])))
        return avg

class Discriminator_img(nn.Module):
    def __init__(self):
        super(Discriminator_img, self).__init__()
        self.res1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.res2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.res3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.res4 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.res5 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.avg_pooling = nn.Sequential(nn.Conv2d(128, 1, 3, 1, 1))
        self.act = nn.Sigmoid()

    def forward(self, x):
        res1 = self.res1(x)
        res2 = self.res2(F.max_pool2d(res1, [2, 2], [2, 2]))
        res3 = self.res3(F.max_pool2d(res2, [2, 2], [2, 2]))
        res4 = self.res4(F.max_pool2d(res3, [2, 2], [2, 2]))
        res5 = self.res5(F.max_pool2d(res4, [2, 2], [2, 2]))
        avg = self.act(self.avg_pooling(F.max_pool2d(res5, [2, 2], [2, 2])))
        return avg

class VGG(nn.Module):
    'Pretrained VGG-19 model features.'
    def __init__(self, layers=(3), replace_pooling = False):
        super(VGG, self).__init__()
        self.layers = layers
        self.instance_normalization = nn.InstanceNorm2d(128)
        self.relu = nn.ReLU()
        self.model = vgg19(pretrained=True).features
        # Changing Max Pooling to Average Pooling
        if replace_pooling:
            self.model._modules['4'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['9'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['18'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['27'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['36'] = nn.AvgPool2d((2,2), (2,2), (1,1))
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for name, layer in enumerate(self.model):
            x = layer(x)
            # if name in self.layers:
            #     features.append(x)
            #     if len(features) == len(self.layers):
            #         break
        return x

#ts = torch.Tensor(16, 3, 64, 64).cuda()
#vr = Variable(ts)
#net = RESCAN().cuda()
#print(net)
#oups = net(vr)
#for oup in oups:
#    print(oup.size())
