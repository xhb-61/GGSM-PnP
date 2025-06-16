import logging
import os.path
from collections import OrderedDict
import sys
import numpy as np
from torch.utils.data import DataLoader
from skimage.io import imsave
sys.path.append("..") 
from utils import utils_option as option
from utils import utils_image as util
from utils import utils_logger

from data.select_dataset import define_Dataset

def gen_logger(opt):
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))
    logger_name = 'pnp'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    return logger

def get_opt(json_path):
    opt = option.parse(json_path, is_train=False)
    opt = option.dict_to_nonedict(opt)
    return opt

def save_opt(opt, pth='sndncnn.json'):
    import json
    with open(pth, 'w') as f:
        json.dump(opt, f)

def get_test_loader(opt):
    dataset_opt = opt['datasets']['test']
    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)
    return test_loader

def eval(model, test_loader, logger):
    idx = 0
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    with open('psnr_ssim_all.txt','w') as f:
            f.write("psnr/ssim:\n") 
    # with open('psnr_ssim_first_avg.txt','w') as f:
    #         f.write("psnr/ssim:\n") 

    psnr_all_avg = np.zeros(model.iter_num)
    ssim_all_avg = np.zeros(model.iter_num)

    print("psnr_all_avg_shape:")
    print(psnr_all_avg.shape)

    for test_data in test_loader:
        idx += 1

        image_name_ext = os.path.basename(test_data['L_path'][0])

        model.feed_data(test_data)

        psnr_all, ssim_all = model.test()

        iter_k = len(psnr_all)

        visuals = model.current_visuals()

        img_E = util.tensor2uint(visuals['E'])          # 256*256
        img_H = util.tensor2uint(visuals['H'])

        out_E_np = img_E
        path1 = 'results/s=%d_p=%d/'%(model.s,model.p)
        os.makedirs(path1,exist_ok=True)
        save_path_E = 'results/s=%d_p=%d/im%d.png'%(model.s,model.p,idx)
        imsave(save_path_E, out_E_np)
        
        # new add psnr_all and ssim_all
        # psnr_all = visuals['psnr_all']
        # ssim_all = visuals['ssim_all']
        
        # print('idx=%d,psnr_all='%(idx),psnr_all)
        # print('idx=%d,ssim_all='%(idx),ssim_all)

        ####### write psnr and ssim for each iteration 
        
        for k in range(iter_k):
            with open('psnr_ssim_all.txt','a') as f:
                f.write("im=%d,iter=%d,psnr_avg=%f,ssim_avg=%f\n"%(idx, k, psnr_all[k], ssim_all[k]))
            psnr_all_avg[k] += psnr_all[k]
            ssim_all_avg[k] += ssim_all[k]
            

        psnr = util.calculate_psnr(img_E, img_H, border=0)
        ssim = util.calculate_ssim(img_E, img_H, border=0)

        logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB; SSIM: {:.4f}'.format(idx, image_name_ext, psnr, ssim))

        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)

        

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('Average PSNR/SSIM - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(ave_psnr, ave_ssim))

    psnr_all_avg = psnr_all_avg/12
    ssim_all_avg = ssim_all_avg/12
    with open('psnr_ssim_first_avg.txt','a') as f:
            f.write("s=%d,p=%d,psnr_avg=%.4f,ssim_avg=%.4f\n"%(model.s, model.p, psnr_all_avg[0], ssim_all_avg[0]))
    
    return ave_psnr, ave_ssim, psnr_all_avg, ssim_all_avg


def eval_avg_only(model, test_loader, logger):
    idx = 0
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    with open('psnr_ssim_avg_only.txt','a') as f:
            # f.write("psnr/ssim:\n") 
            f.write("s=%d,p=%d:  "%(model.s,model.p))

    # psnr_all_avg = np.zeros([12,model.iter_num])
    # ssim_all_avg = np.zeros([12,model.iter_num])
    for test_data in test_loader:
        idx += 1

        image_name_ext = os.path.basename(test_data['L_path'][0])

        model.feed_data(test_data)

        psnr_all, ssim_all = model.test()
        iter_k = len(psnr_all)

        visuals = model.current_visuals()

        img_E = util.tensor2uint(visuals['E'])
        img_H = util.tensor2uint(visuals['H'])

        # new add psnr_all and ssim_all
        # psnr_all = visuals['psnr_all']
        # ssim_all = visuals['ssim_all']
        
        # print('idx=%d,psnr_all='%(idx),psnr_all)
        # print('idx=%d,ssim_all='%(idx),ssim_all)

        ####### write psnr and ssim for each iteration 
        
        # for k in range(iter_k):
        #     with open('psnr_ssim_all.txt','a') as f:
        #         f.write("im=%d,iter=%d,psnr_avg=%f,ssim_avg=%f\n"%(idx, k, psnr_all[k], ssim_all[k]))

        psnr = util.calculate_psnr(img_E, img_H, border=0)
        ssim = util.calculate_ssim(img_E, img_H, border=0)

        logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB; SSIM: {:.4f}'.format(idx, image_name_ext, psnr, ssim))

        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)

        # psnr_all_avg[idx-1,:] += psnr_all
        # ssim_all_avg[idx-1,:] += ssim_all

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('Average PSNR/SSIM - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(ave_psnr, ave_ssim))

    with open('psnr_ssim_avg_only.txt','a') as f:
            f.write("psnr=%.2f,ssim=%.4f:\n"%(ave_psnr,ave_ssim)) 
    
    return ave_psnr, ave_ssim

def eval_diff_q(model, test_loader, logger):
    idx = 0
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    # with open('psnr_ssim_all_diff_q.txt','w') as f:
    #         f.write("psnr/ssim:\n") 
    # with open('psnr_ssim_first_avg.txt','w') as f:
    #         f.write("psnr/ssim:\n") 

    psnr_all_avg = np.zeros(model.iter_num)
    ssim_all_avg = np.zeros(model.iter_num)

    # print("psnr_all_avg_shape:")
    # print(psnr_all_avg.shape)

    for test_data in test_loader:
        idx += 1

        image_name_ext = os.path.basename(test_data['L_path'][0])

        model.feed_data(test_data)

        psnr_all, ssim_all, q = model.test()

        iter_k = len(psnr_all)

        visuals = model.current_visuals()

        img_E = util.tensor2uint(visuals['E'])          # 256*256
        img_H = util.tensor2uint(visuals['H'])

        out_E_np = img_E
        path1 = 'results_diff_q1/s=%d_p=%d,q=%.2f/'%(model.s,model.p,q)
        os.makedirs(path1,exist_ok=True)
        save_path_E = 'results_diff_q1/s=%d_p=%d,q=%.2f/im%d.png'%(model.s,model.p,q,idx)
        imsave(save_path_E, out_E_np)
        
        # new add psnr_all and ssim_all
        # psnr_all = visuals['psnr_all']
        # ssim_all = visuals['ssim_all']
        
        # print('idx=%d,psnr_all='%(idx),psnr_all)
        # print('idx=%d,ssim_all='%(idx),ssim_all)

        ####### write psnr and ssim for each iteration 
        
        for k in range(iter_k):
            with open('psnr_ssim_all_diff_q.txt','a') as f:
                f.write("im=%d,iter=%d,psnr_avg=%f,ssim_avg=%f,q=%.2f\n"%(idx, k, psnr_all[k], ssim_all[k],q))
            psnr_all_avg[k] += psnr_all[k]
            ssim_all_avg[k] += ssim_all[k]
            

        psnr = util.calculate_psnr(img_E, img_H, border=0)
        ssim = util.calculate_ssim(img_E, img_H, border=0)

        print("q=%.2f"%q)

        logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB; SSIM: {:.4f}'.format(idx, image_name_ext, psnr, ssim))

        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)

        

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('Average PSNR/SSIM - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(ave_psnr, ave_ssim))

    psnr_all_avg = psnr_all_avg/12
    ssim_all_avg = ssim_all_avg/12
    with open('psnr_ssim_first_avg.txt','a') as f:
            f.write("s=%d,p=%d,psnr_avg=%.4f,ssim_avg=%.4f\n"%(model.s, model.p, psnr_all_avg[0], ssim_all_avg[0]))
    
    return ave_psnr, ave_ssim, psnr_all_avg, ssim_all_avg