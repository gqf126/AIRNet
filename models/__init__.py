import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
    elif model == 'srragan':
        from .SRRaGAN_model import SRRaGANModel as M
    elif model == 'sftgan':
        from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
    elif model == 'image_para':
        from .Image_Para_model import Image_Para_model as M
    elif model == 'image_Y_UV_para':
        from .Image_Y_UV_Para_model import Image_Y_UV_Para_model as M
    elif model == 'image_Y_Gamma_UV_para':
        from .Image_Y_Gamma_UV_Para_model import Image_Y_Gamma_UV_Para_model as M        
    elif model == 'image_Y_UV_para_test':
        from .Image_Y_UV_Para_test_model import Image_Y_UV_Para_test_model as M   
    elif model == 'image_Y_UV_boosted_para_test':
        from .Image_Y_UV_boosted_Para_test_model import Image_Y_UV_boosted_Para_test_model as M             
    elif model == 'image_Y_UV_para_gan':
        from .Image_Y_UV_Para_GAN_model import Image_Y_UV_Para_GAN_model as M              
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
