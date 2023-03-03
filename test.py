import os
import sys
import logging
import time
import argparse
import numpy as np
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import scipy.io as sio

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)

util.setup_logger(None, opt['path']['log'], 'test.log', level=logging.INFO, screen=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))
# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    para_list = {}
    for data in test_loader:
        # time_start = time.time()

        need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True
        model.feed_data(data, need_HR=need_HR)
        img_path = data['LR_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        model_inference_start_time = time.time()
        model.test()  # test
        # model_inference_time = time.time() - model_inference_start_time

        # img_process_start_time = time.time()
        visuals = model.get_current_visuals(need_HR=need_HR)

        # lr_ori_img = util.tensor2img(visuals['LR_ori'], color_type='ycbcr_matlab')  # uint8
        # sr_img = util.tensor2img(visuals['SR'], color_type='ycbcr')  # uint8
        # sr_img = util.tensor2img(visuals['SR'], color_type='lab_hdr', out_type=np.float32)  # uint8
        sr_img = util.tensor2img(visuals['SR'], color_type='lab')  # uint8
        # sr_img = util.tensor2img(visuals['SR'], color_type='ycbcr_matlab')  # uint8
        # img_process_time = time.time() - img_process_start_time
        # img_process_start_time = time.time()

        # para_list.update({str(img_name[:5]): [{'para_y': para_y}, {'para_y_boosted': para_y_boosted},
        #     {'para_uv': para_uv}, {'para_uv_boosted': para_uv_boosted}]})
        # save imagess
        suffix = opt['suffix']
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_img_path = os.path.join(dataset_dir, img_name + '.png')
        # util.save_img_hdr(sr_img, save_img_path)
        util.save_img(sr_img, save_img_path)
        # img_process_time = time.time() - img_process_start_time

        # util.save_img(np.concatenate((lr_ori_img, sr_img), 1), save_img_path)
        # print("process time per img: {}, modle inference time: {}, img process time: {}".format((time.time()-time_start)/visuals['SR'].shape[0],
        #     model_inference_time/visuals['SR'].shape[0], img_process_time/visuals['SR'].shape[0]))
