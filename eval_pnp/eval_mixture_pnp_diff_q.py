import argparse
import sys
import numpy as np
sys.path.append(".") 

from models.select_model import define_Model
from pnp.util_pnp import get_test_loader, eval, eval_diff_q, get_opt, gen_logger

def unpack_opt(opt):
    opt['datasets']['test']['sigma'] = opt['sigma']
    opt['datasets']['test']['sigma_test'] = opt['sigma']
    opt['datasets']['test']['sp'] = opt['sp']
    test_loader = get_test_loader(opt)
    opt['netG']['sigma'] = opt['sigma']
    model = define_Model(opt)
    return model, test_loader

def _get_json_path():
    json_path = 'options/pnp/mixture_diff_q.json'
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')
    json_path = parser.parse_args().opt
    return json_path


json_path = _get_json_path()
opt = get_opt(json_path)


for i in range(1):
    for j in range(1):
        for qq0 in range(20):
        # print(opt['netG']['beta'])
            # opt['sigma'] =  (i+1)*10
            # opt['sp'] = (j+1)*10
            # opt['datasets']['test']['dataroot_L'] = './testsets/set12_noise/%d_%d'%((i+1)*10,(j+1)*10)
            # opt['datasets']['test']['dataroot_C'] = './testsets/set12_deblur/%d_%.1f'%((i+1)*10,(j+1)/10.0)
            opt['q'] = qq0 * 0.1 + 0.1
            logger = gen_logger(opt)
            # with open('psnr_ssim_all_diff_q.txt','w') as f:
            #     f.write("psnr/ssim:\n") 
            eval_diff_q(*unpack_opt(opt), logger)