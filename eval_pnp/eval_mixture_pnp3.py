import argparse
import sys
import numpy as np
sys.path.append(".") 

from models.select_model import define_Model
from pnp.util_pnp import get_test_loader, eval, get_opt, gen_logger

def unpack_opt(opt):
    opt['datasets']['test']['sigma'] = opt['sigma']
    opt['datasets']['test']['sigma_test'] = opt['sigma']
    opt['datasets']['test']['sp'] = opt['sp']
    test_loader = get_test_loader(opt)
    opt['netG']['sigma'] = opt['sigma']
    model = define_Model(opt)
    return model, test_loader

def _get_json_path():
    json_path = 'options/pnp/mixture.json'
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')
    json_path = parser.parse_args().opt
    return json_path


json_path = _get_json_path()
opt = get_opt(json_path)


for i in range(5):
    for j in range(5):
        # print(opt['netG']['beta'])
        opt['sigma'] =  (i+1)*10
        opt['sp'] = (j+1)*10
        opt['datasets']['test']['dataroot_L'] = './testsets/set12_noise/%d_%d'%((i+1)*10,(j+1)*10)
        opt['datasets']['test']['dataroot_C'] = './testsets/set12_deblur/%d_%.1f'%((i+1)*10,(j+1)/10.0)
        logger = gen_logger(opt)
        eval(*unpack_opt(opt), logger)
        