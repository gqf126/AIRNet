import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
import scipy.io as sio


class LRHR_Meta_Y_UV_test_Dataset(data.Dataset):
    '''
    Read HR image, segmentation probability map; generate LR image, category for SFTGAN
    also sample general scenes for background
    need to generate LR images on-the-fly
    '''

    def __init__(self, opt):
        super(LRHR_Meta_Y_UV_test_Dataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.paths_HR = None
        self.paths_HR_bg = None  # HR images for background scenes
        self.LR_env = None  # environment for lmdb
        self.HR_env = None
        self.HR_env_bg = None
        # print('*********',opt['data_type'],opt['dataroot_HR_bg'])
        # read image list from lmdb or image files
        self.HR_env, self.paths_HR = util.get_image_paths(opt['data_type'], opt['dataroot_HR'])
        self.LR_env, self.paths_LR = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])
        self.HR_env_bg, self.paths_HR_bg = util.get_image_paths(opt['data_type'], \
            opt['dataroot_HR_bg'])
        #self.paths_HR = self.paths_HR[:128]
        #self.paths_LR = self.paths_LR[:128]

        assert self.paths_HR, 'Error: HR path is empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                'HR and LR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_HR))

        self.random_scale_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
        self.ratio = 10  # 10 OST data samples and 1 DIV2K general data samples(background)
        self.scale = self.opt['scale']
        self.HR_size = 128

    def __getitem__(self, index):
        HR_path, LR_path = None, None
        # get HR image
        if self.opt['phase'] == 'train' and \
                random.choice(list(range(self.ratio))) == 20:  # read background images
            bg_index = random.randint(0, len(self.paths_HR_bg) - 1)
            HR_path = self.paths_HR_bg[bg_index]
            img_HR = util.read_img(self.HR_env_bg, HR_path)
            seg = torch.FloatTensor(8, img_HR.shape[0], img_HR.shape[1]).fill_(0)
            seg[0, :, :] = 1  # background
        else:
            HR_path = self.paths_HR[index]
            LR_path = self.paths_LR[index]
            img_HR = util.read_img(self.HR_env, HR_path)
            # seg = torch.load(HR_path.replace('/img/', '/bicseg/').replace('.png', '.pth'))
            # cseg = sio.loadmat(LR_path.replace('/img/', '/bicseg/').replace('.tif', '.mat'))
            # seg = cseg['classseg']
            # read segmentatin files, you should change it to your settings.

        # modcrop in the validation / test phase
        # if self.opt['phase'] != 'train':
        #     img_HR = util.modcrop(img_HR, 1)

        # seg = np.transpose(seg.numpy(), (1, 2, 0))
        # seg = np.transpose(seg, (1, 2, 0))
        # get LR image
        if self.paths_LR:
            img_LR = util.read_img(self.LR_env, LR_path)
            # if self.opt['phase'] != 'train':
            #     hh, ww, _ = img_HR.shape
            #     r_h = random.randint(0, max(0, hh-self.patch_size))
            #     r_w = random.randint(0, max(0, ww-self.patch_size))
            #     img_HR = img_HR[r_h : r_h + self.patch_size, r_w : r_w + self.patch_size, :]
            #     img_LR = img_LR[r_h : r_h + self.patch_size, r_w : r_w + self.patch_size, :]
            #     seg = seg[r_h : r_h + self.patch_size, r_w : r_w + self.patch_size, :]
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = seg.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // self.scale) * self.scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, self.scale, self.HR_size)
                W_s = _mod(W_s, random_scale, self.scale, self.HR_size)
                img_HR = cv2.resize(np.copy(img_HR), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                # seg = cv2.resize(np.copy(seg), (W_s, H_s), interpolation=cv2.INTER_NEAREST)

            H, W, _ = img_HR.shape
            # using matlab imresize
            img_LR = util.imresize_np(img_HR, 1 / self.scale, True)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        H, W, C = img_LR.shape
        # if self.opt['phase'] in ['train', 'val']:
        #     LR_size = self.HR_size // self.scale

        #     # randomly crop
        #     rnd_h = random.randint(0, max(0, H - LR_size))
        #     rnd_w = random.randint(0, max(0, W - LR_size))
        #     img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
        #     rnd_h_HR, rnd_w_HR = int(rnd_h * self.scale), int(rnd_w * self.scale)
        #     img_HR = img_HR[rnd_h_HR:rnd_h_HR + self.HR_size, rnd_w_HR:rnd_w_HR + self.HR_size, :]
        #     seg = seg[rnd_h_HR:rnd_h_HR + self.HR_size, rnd_w_HR:rnd_w_HR + self.HR_size, :]
        #     # seg_label = seg.mean((0,1))
        #     # seg_label[seg_label > 0] = 1
        #     # augmentation - flip, rotate
        #     img_LR, img_HR, seg = util.augment([img_LR, img_HR, seg], self.opt['use_flip'],
        #                                        self.opt['use_rot'])
        if 'test' in self.opt['phase']:
            LR_size = self.HR_size // self.scale
            assert self.HR_size == min([H, W])
            # print(LR_size, img_LR.shape)
            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            img_LR_small = util.imresize_np(img_LR, 0.5)
            # seg = seg[rnd_h:rnd_h + self.HR_size, rnd_w:rnd_w + self.HR_size, :]
            # print(LR_size, img_LR.shape, img_HR.shape)
            # category
        #     if 'building' in HR_path:
        #         category = 1
        #     elif 'plant' in HR_path:
        #         category = 2
        #     elif 'mountain' in HR_path:
        #         category = 3
        #     elif 'water' in HR_path:
        #         category = 4
        #     elif 'sky' in HR_path:
        #         category = 5
        #     elif 'grass' in HR_path:
        #         category = 6
        #     elif 'animal' in HR_path:
        #         category = 7
        #     else:
        #         category = 0  # background
        # else:
        #     category = -1  # during val, useless

        # BGR2Y, output 3 channels
        # a = img_HR.copy()
        # print(a.shape, np.max(a), np.min(a))
        img_LR, img_HR, img_LR_small = util.channel_convert(in_c=img_LR.shape[2], tar_type='lab', \
           img_list=[img_LR, img_HR, img_LR_small])
        # img_LR, img_HR, img_LR_small = util.channel_convert(in_c=img_LR.shape[2], tar_type='ycbcr_matlab', \
        #     img_list=[img_LR, img_HR, img_LR_small])

        # print(img_HR.dtype, np.max(img_HR), np.min(img_HR))
        # b = util.ycbcr2bgr(img_HR)
        # print(b.shape, np.max(b), np.min(b))
        # print(a[:10,10,0],b[:10,10,2])
        # fa
        # print(np.mean(a-b))
        # fds

        img_LR_y, img_HR_y = [np.tile(temp, 3) for temp in [img_LR[:,:,0:1], img_HR[:,:,0:1]]]
        img_LR_uv, img_HR_uv = img_LR[:,:,1:], img_HR[:,:,1:]
        img_LR_uv = img_LR_small[:,:,1:]

        # BGR to RGB, HWC to CHW, numpy to tensor
        # if img_HR_y.shape[2] == 3 or img_HR_uv.shape[2] == 3:
        #     img_HR_y = img_HR_y[:, :, [2, 1, 0]]
        #     img_HR_uv = img_HR_uv[:, :, [2, 1, 0]]
        #     img_LR_y = img_LR_y[:, :, [2, 1, 0]]
        #     img_LR_uv = img_LR_uv[:, :, [2, 1, 0]]
        img_HR_y = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR_y, (2, 0, 1)))).float()
        img_HR_uv = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR_uv, (2, 0, 1)))).float()
        img_LR_y = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR_y, (2, 0, 1)))).float()
        img_LR_uv = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR_uv, (2, 0, 1)))).float()
        # seg = torch.from_numpy(np.ascontiguousarray(np.transpose(seg, (2, 0, 1)))).float()
        # seg_label = torch.from_numpy(np.ascontiguousarray(seg_label)).float()
        if LR_path is None:
            LR_path = HR_path
        return {
            'LR_y': img_LR_y,
            'LR_uv': img_LR_uv,
            'HR_y': img_HR_y,
            'HR_uv': img_HR_uv,
            # 'seg': seg,
            # 'seg_label': seg_label,
            # 'category': category,
            'LR_path': LR_path,
            'HR_path': HR_path
        }

    def __len__(self):
        return len(self.paths_HR)
