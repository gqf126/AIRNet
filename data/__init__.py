'''create dataset and dataloader'''
import logging
import torch.utils.data


def create_dataloader(dataset, dataset_opt):
    '''create dataloader '''
    phase = dataset_opt['phase']
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['n_workers'],
            drop_last=True,
            pin_memory=True)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=dataset_opt['batch_size'], shuffle=False, num_workers=dataset_opt['n_workers'], pin_memory=True)


def create_dataset(dataset_opt):
    '''create dataset'''
    mode = dataset_opt['mode']
    if mode == 'LR':
        from data.LR_dataset import LRDataset as D
    elif mode == 'LRHR':
        from data.LRHR_dataset import LRHRDataset as D
    elif mode == 'LRHRseg_bg':
        from data.LRHR_seg_bg_dataset import LRHRSeg_BG_Dataset as D
    elif mode == 'LRHR_seg_bg_yuv':
        from data.LRHR_seg_bg_yuv_dataset import LRHRSeg_BG_yuv_Dataset as D
    elif mode == 'LRHR_Y_UV':
        from data.LRHR_Y_UV_dataset import LRHR_Y_UV_Dataset as D
    elif mode == 'LRHR_Y_UV_para':
        from data.LRHR_Y_UV_para_dataset import LRHR_Y_UV_para_Dataset as D
    elif mode == 'LRHR_Meta_Y_UV':
        from data.LRHR_Meta_Y_UV_dataset import LRHR_Meta_Y_UV_Dataset as D
    elif mode == 'LRHR_Meta_Y_UV_test':
        from data.LRHR_Meta_Y_UV_test_dataset import LRHR_Meta_Y_UV_test_Dataset as D
    elif mode == 'LRHR_Meta_Y_UV_boosted_test':
        from data.LRHR_Meta_Y_UV_boosted_test_dataset import LRHR_Meta_Y_UV_boosted_test_Dataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
