import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss, GradientPenaltyLoss
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm

logger = logging.getLogger('base')


class Image_Y_UV_Para_test_model(BaseModel):
    def __init__(self, opt):
        super(Image_Y_UV_Para_test_model, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG_Y = networks.define_G_Y(opt).to(self.device)  # G
        self.netG_UV = networks.define_G_UV(opt).to(self.device)  # G
        if self.is_train:
            # self.netD = networks.define_D(opt).to(self.device)  # D
            self.netG_Y.train()
            self.netG_UV.train()
            # self.netD.train()
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logging.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logging.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_G_Y_params_fea = []
            optim_G_Y_params_cls = []
            for k, v in self.netG_Y.named_parameters():  # can optimize for a part of the model
                if 'cls' in k or '6' in k or '7' in k:
                    # print('cls weight', k)
                    optim_G_Y_params_cls.append(v)
                else:
                    # print('fea weight', k)
                    optim_G_Y_params_fea.append(v)
            self.optimizer_G_Y = torch.optim.SGD([
                {'params': optim_G_Y_params_fea},
                {'params': optim_G_Y_params_cls, 'lr': 10*train_opt['lr_G_Y']}
            ], lr=train_opt['lr_G_Y'], momentum=0.9)
            self.optimizers.append(self.optimizer_G_Y)

            optim_G_UV_params = []
            for k, v in self.netG_UV.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_G_UV_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
                    
            self.optimizer_G_UV = torch.optim.SGD(optim_G_UV_params, lr=train_opt['lr_G_UV'], momentum=0.9)
            self.optimizers.append(self.optimizer_G_UV)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        # print network
        self.print_network()

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L_y = data['LR_y'].to(self.device)
        self.var_L_uv = data['LR_uv'].to(self.device)

        self.path = data['HR_path']
        if need_HR:  # train or val
            self.var_H_y = data['HR_y'].to(self.device)
            self.var_H_uv = data['HR_uv'].to(self.device)
        try:
            self.var_L_ori = [temp.to(self.device) for temp in data['img_LR_ori']]
        except:
            pass

    def optimize_parameters(self, step):
        # G
        self.optimizer_G_Y.zero_grad()
        self.optimizer_G_UV.zero_grad()
        
        self.para_y = self.netG_Y(self.var_L_y)
        self.para_uv = self.netG_UV((self.var_L_uv, self.var_seg)).view(self.var_L_uv.shape[0], 2, 3)
        temp_tensor1 = torch.tensor([0,0,1]).float().repeat(self.var_L_uv.shape[0], 1, 1).to(self.device)
        para_uv_transform = torch.cat((self.para_uv, temp_tensor1), 1).to(self.device)
        var_L_uv_flatten = self.var_L_uv.view(self.var_L_uv.shape[0], 2, -1)
        temp_tensor2 = torch.ones(self.var_L_uv.shape[0], 1, var_L_uv_flatten.shape[2]).float().to(self.device)
        var_L_uv_1 = torch.cat((var_L_uv_flatten, temp_tensor2), 1)

        # 4th polynomial
        self.fake_H_y = self.para_y[:,0].view(self.var_L_y.shape[0],1,1,1) * torch.pow(self.var_L_y, 4) + \
                      self.para_y[:,1].view(self.var_L_y.shape[0],1,1,1) * torch.pow(self.var_L_y, 3) + \
                      self.para_y[:,2].view(self.var_L_y.shape[0],1,1,1) * torch.pow(self.var_L_y, 2) + \
                      self.para_y[:,3].view(self.var_L_y.shape[0],1,1,1) * self.var_L_y + \
                      self.para_y[:,4].view(self.var_L_y.shape[0],1,1,1) * torch.ones_like(self.var_L_y)

        self.fake_H_uv = torch.matmul(para_uv_transform, var_L_uv_1)[:, :2, :]
        self.fake_H_uv = self.fake_H_uv.view_as(self.var_L_uv)

        fake_H = torch.cat((self.fake_H_y[:,0:1,:,:], self.fake_H_uv), 1)
        var_H = torch.cat((self.var_H_y[:,0:1,:,:], self.var_H_uv), 1)
        l_g_total = 0
        # if step % self.D_update_ratio == 0 and step > self.D_init_iters:
        if self.cri_pix:  # pixel loss
            # l_g_y_pix = self.l_pix_w * self.cri_pix(self.fake_H_y, self.var_H_y)
            # l_g_total += l_g_y_pix
            l_g_pix = self.l_pix_w * self.cri_pix(fake_H, var_H)
            l_g_total += l_g_pix
        if self.cri_fea:  # feature loss
            real_fea = self.netF(self.var_H).detach()
            fake_fea = self.netF(self.fake_H)
            l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
            l_g_total += l_g_fea
        # G gan + cls loss
        # pred_g_fake, cls_g_fake = self.netD(self.fake_H)
        # pred_g_fake = self.netD(self.fake_H)
        # pred_g_fake = self.netD(self.fake_H)
        # l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
        # l_g_cls = self.l_gan_w * self.cri_ce(cls_g_fake, self.var_label)
        # l_g_total += l_g_gan
        # l_g_total += l_g_cls

        l_g_total.backward()
        self.optimizer_G_Y.step()
        self.optimizer_G_UV.step()

        # set log
        # if step % self.D_update_ratio == 0 and step > self.D_init_iters:
        # G

        if self.cri_pix:
            self.log_dict['l_g_pix'] = l_g_pix.item()
        if self.cri_fea:
            self.log_dict['l_g_fea'] = l_g_fea.item()

    def test(self):
        self.netG_Y.eval()
        self.netG_UV.eval()
        with torch.no_grad():
            self.para_y = self.netG_Y(self.var_L_y)

            # 4th polynomial
            # self.fake_H_y = self.para_y[:,0].view(self.var_H_y.shape[0],1,1,1) * torch.pow(self.var_H_y, 4) + \
            #               self.para_y[:,1].view(self.var_H_y.shape[0],1,1,1) * torch.pow(self.var_H_y, 3) + \
            #               self.para_y[:,2].view(self.var_H_y.shape[0],1,1,1) * torch.pow(self.var_H_y, 2) + \
            #               self.para_y[:,3].view(self.var_H_y.shape[0],1,1,1) * self.var_H_y + \
            #               self.para_y[:,4].view(self.var_H_y.shape[0],1,1,1) * torch.ones_like(self.var_H_y)

            self.fake_H_y = self.para_y[:,0].view(self.var_H_y.shape[0],1,1,1) * torch.pow(self.var_H_y, 4) + \
                                      self.para_y[:,1].view(self.var_H_y.shape[0],1,1,1) * torch.pow(self.var_H_y, 3) + \
                                      self.para_y[:,2].view(self.var_H_y.shape[0],1,1,1) * torch.pow(self.var_H_y, 2) + \
                                      self.para_y[:,3].view(self.var_H_y.shape[0],1,1,1) * self.var_H_y

            self.para_uv = self.netG_UV(self.var_L_uv).view(self.var_H_uv.shape[0], 2, 3)
            # self.para_uv = self.netG_UV((self.var_L_uv, self.var_seg)).view(self.var_H_uv.shape[0], 2, 3)
            temp_tensor1 = torch.tensor([0, 0, 1]).float().repeat(self.var_H_uv.shape[0], 1, 1).to(self.device)
            para_uv_transform = torch.cat((self.para_uv, temp_tensor1), 1).to(self.device)

            var_H_uv_flatten = self.var_H_uv.view(self.var_H_uv.shape[0], 2, -1)
            temp_tensor2 = torch.ones(self.var_H_uv.shape[0], 1, var_H_uv_flatten.shape[2]).float().to(self.device)
            var_H_uv_1 = torch.cat((var_H_uv_flatten, temp_tensor2), 1)

            self.fake_H_uv = torch.matmul(para_uv_transform, var_H_uv_1)[:, :2, :]
            self.fake_H_uv = self.fake_H_uv.view_as(self.var_H_uv)

        self.netG_Y.train()
        self.netG_UV.train()

    def test2(self):
        self.netG_Y.eval()
        self.netG_UV.eval()
        with torch.no_grad():
            self.para_y = self.netG_Y(self.var_L_y)
            # 4th polynomial
            self.fake_H_y = self.para_y[:,0].view(self.var_L_ori[0].shape[0],1,1,1) * torch.pow(self.var_L_ori[0], 4) + \
                          self.para_y[:,1].view(self.var_L_ori[0].shape[0],1,1,1) * torch.pow(self.var_L_ori[0], 3) + \
                          self.para_y[:,2].view(self.var_L_ori[0].shape[0],1,1,1) * torch.pow(self.var_L_ori[0], 2) + \
                          self.para_y[:,3].view(self.var_L_ori[0].shape[0],1,1,1) * self.var_L_ori[0] + \
                          self.para_y[:,4].view(self.var_L_ori[0].shape[0],1,1,1) * torch.ones_like(self.var_L_ori[0])
            self.para_uv = self.netG_UV(self.var_L_uv).view(self.var_L_ori[1].shape[0], 2, 3)
            temp_tensor1 = torch.tensor([0, 0, 1]).float().repeat(self.var_L_ori[1].shape[0], 1, 1).to(self.device)
            para_uv_transform = torch.cat((self.para_uv, temp_tensor1), 1).to(self.device)

            var_L_uv_flatten = self.var_L_ori[1].view(self.var_L_ori[1].shape[0], 2, -1)
            temp_tensor2 = torch.ones(self.var_L_ori[1].shape[0], 1, var_L_uv_flatten.shape[2]).float().to(self.device)
            var_L_uv_1 = torch.cat((var_L_uv_flatten, temp_tensor2), 1)

            self.fake_H_uv = torch.matmul(para_uv_transform, var_L_uv_1)[:, :2, :]
            self.fake_H_uv = self.fake_H_uv.view_as(self.var_L_ori[1])

        self.netG_Y.train()
        self.netG_UV.train()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        # out_dict['LR'] = torch.cat((self.var_L_y.detach()[0, 0:1].float().cpu(),
        #                             self.var_L_uv.detach()[0].float().cpu()), 0)

        # out_dict['LR_ori'] = torch.cat((self.var_L_ori[0].detach()[0, 0:1].float().cpu(),
        #                                 self.var_L_ori[1].detach()[0].float().cpu()), 0)
        # test iterative, fit uv
        out_dict['SR'] = torch.cat((self.fake_H_y.detach()[0, 0:1].float().cpu(),
                                    self.fake_H_uv.detach()[0].float().cpu()), 0)
        # out_dict['SR'] = torch.cat((self.fake_H_y.detach()[0, 0:1].float().cpu(),
        #                             self.fake_H_uv.detach()[0].float().cpu()), 0)
        out_dict['SR_all'] = torch.cat((self.fake_H_y.detach()[:, 0:1].float().cpu(),
                                    self.fake_H_uv.detach().float().cpu()), 1)
        if need_HR:
            out_dict['HR'] = torch.cat((self.var_H_y.detach()[0, 0:1].float().cpu(),
                                        self.var_H_uv.detach()[0].float().cpu()), 0)
            out_dict['HR_all'] = torch.cat((self.var_H_y.detach()[:, 0:1].float().cpu(),
                                        self.var_H_uv.detach().float().cpu()), 1)
            out_dict['HR_path'] = self.path
        return out_dict

    def print_network(self):
        # G_Y
        s, n = self.get_network_description(self.netG_Y)
        if isinstance(self.netG_Y, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG_Y.__class__.__name__,
                                             self.netG_Y.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG_Y.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)
        # G_UV
        s, n = self.get_network_description(self.netG_UV)
        if isinstance(self.netG_UV, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG_UV.__class__.__name__,
                                             self.netG_UV.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG_UV.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)
        if self.is_train:
            # D
            # s, n = self.get_network_description(self.netD)
            # if isinstance(self.netD, nn.DataParallel):
            #     net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
            #                                     self.netD.module.__class__.__name__)
            # else:
            #     net_struc_str = '{}'.format(self.netD.__class__.__name__)

            # logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            # logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                    self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)

                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_G_Y = self.opt['path']['pretrain_model_G_Y']
        if load_path_G_Y is not None:
            logger.info('Loading pretrained model for G_Y [{:s}] ...'.format(load_path_G_Y))
            self.load_network(load_path_G_Y, self.netG_Y)
            print('load pretrained_model_G_Y')
        load_path_G_UV = self.opt['path']['pretrain_model_G_UV']
        if load_path_G_UV is not None:
            logger.info('Loading pretrained model for G_UV [{:s}] ...'.format(load_path_G_UV))
            self.load_network(load_path_G_UV, self.netG_UV)
            print('load pretrained_model_G_UV')
        # load_path_D = self.opt['path']['pretrain_model_D']
        # if self.opt['is_train'] and load_path_D is not None:
        #     logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
        #     self.load_network(load_path_D, self.netD)

    def save(self, iter_step):
        self.save_network(self.netG_Y, 'G_Y', iter_step)
        self.save_network(self.netG_UV, 'G_UV', iter_step)
        # self.save_network(self.netD, 'D', iter_step)
