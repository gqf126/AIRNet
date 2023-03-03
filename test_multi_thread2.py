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
import threading
from multiprocessing import Process, Pool

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
threads_for_save_img = 32
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
        time_start = time.time()

        need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True
        model.feed_data(data, need_HR=need_HR)
        img_path = data['LR_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        model_inference_start_time = time.time()
        model.test()  # test
        model_inference_time = time.time() - model_inference_start_time

        img_process_start_time = time.time()
        time_tensor_start = time.time()
        visuals = model.get_current_visuals(need_HR=need_HR)

        thread_list = []
        batch_size = visuals['SR_all'].shape[0]
        # print(visuals['SR_all'].device)
        # print(visuals['HR_path'].device)
        # fds
        imgs_path = visuals['HR_path']
        # imgs = visuals['SR_all']
        #time0 = time.time()
        imgs = visuals['SR_all'].float().cpu().numpy()
        time_tensor = time.time() - time_tensor_start
        time1 = time.time()
        for i in range(threads_for_save_img):
            temp_start = i*batch_size//threads_for_save_img
            temp_end = (i+1)*batch_size//threads_for_save_img
            t = threading.Thread(target=util.save_imgs_in_batch, args=(imgs[temp_start:temp_end],
                                    dataset_dir, imgs_path[temp_start:temp_end] ))
            thread_list.append(t)
        time2 = time.time()
        time_thread1 = time2 - time1
        for t in thread_list:
            t.start()

        for t in thread_list:
            t.join()
        time_thread2 = time.time() - time2

        img_process_time = time.time() - img_process_start_time

        # process_list = []
        # # pool = Pool(threads_for_save_img)
        # for i in range(threads_for_save_img):
        #     temp_start = i*batch_size//threads_for_save_img
        #     temp_end = (i+1)*batch_size//threads_for_save_img
        #     p = Process(target=util.save_imgs_in_batch, args=(visuals['SR_all'][temp_start:temp_end],
        #                             dataset_dir, visuals['HR_path'][temp_start:temp_end]))
        #     p.start()
        #     process_list.append(p)

        # for p in process_list:
        #     p.join()

        # for i in range(threads_for_save_img):
        #     temp_start = i*batch_size//threads_for_save_img
        #     temp_end = (i+1)*batch_size//threads_for_save_img
        #     pool.apply_async(func=util.save_imgs_in_batch, args=(visuals['SR_all'][temp_start:temp_end],
        #                             dataset_dir, visuals['HR_path'][temp_start:temp_end],))

        # pool.close()
        # pool.join()
        # print(batch_size)
        # print(visuals['SR'].shape[0])
        # print(visuals['SR_all'].shape[0])
        print("process time per img: {}, modle inference time: {}, img process time: {}, time tensor: {},  thread1: {}, thread2: {}".format((time.time()-time_start)/batch_size,
            model_inference_time/batch_size, img_process_time/batch_size, time_tensor, time_thread1, time_thread2))
        # # lr_ori_img = util.tensor2img(visuals['LR_ori'], color_type='ycbcr_matlab')  # uint8
        # # sr_img = util.tensor2img(visuals['SR'], color_type='ycbcr')  # uint8
        # # sr_img = util.tensor2img(visuals['SR'], color_type='lab_hdr', out_type=np.float32)  # uint8
        # # sr_img = util.tensor2img(visuals['SR'], color_type='lab')  # uint8
        # sr_img = util.tensor2img(visuals['SR'], color_type='ycbcr_matlab')  # uint8

        # # para_list.update({str(img_name[:5]): [{'para_y': para_y}, {'para_y_boosted': para_y_boosted},
        # #     {'para_uv': para_uv}, {'para_uv_boosted': para_uv_boosted}]})
        # # save imagess
        # suffix = opt['suffix']
        # if suffix:
        #     save_img_path = os.path.join(dataset_dir, img_name + suffix + '.png')
        # else:
        #     save_img_path = os.path.join(dataset_dir, img_name + '.png')
        # # util.save_img_hdr(sr_img, save_img_path)
        # util.save_img(sr_img, save_img_path)
        # # util.save_img(np.concatenate((lr_ori_img, sr_img), 1), save_img_path)
