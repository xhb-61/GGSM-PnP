import torch
from torch import nn
import utils.utils_image as util
from pnp.mixture_admm import *
import os
from skimage.io import imsave

def get_psnr_ssim(u1, img_H):
        pre_i = torch.clamp(u1 / 255, 0., 1.)
        img_E = util.tensor2uint(pre_i)
        img_H = util.tensor2uint(img_H)
        psnr = util.calculate_psnr(img_E, img_H, border=0)
        ssim = util.calculate_ssim(img_E, img_H, border=0)
        return psnr, ssim

class Intermediate:
    def __init__(self, GT):
        self.available = not GT is None
        if self.available:
            self.GT = GT
            self.best_psnr_result = None
            self.best_ssim_result = None
            self.best_measure_result = None
            self.max_psnr = 0
            self.max_ssim = 0
            self.max_measure = 0
    
    def _measure_psnr_ssim(self, psnr, ssim):
        measure = psnr + 10*ssim
        return measure

    def is_available(self):
        return self.available

    def update(self, u):
        if not self.is_available():
            return None

        cur_psnr, cur_ssim = self._get_intermediate_results(u, self.GT)

        if self.max_psnr < cur_psnr:
            self.max_psnr = cur_psnr
            self.best_psnr_result = u   

        if self.max_ssim < cur_ssim:
            self.max_ssim = cur_ssim
            self.best_ssim_result = u

        cur_measure = self._measure_psnr_ssim(cur_psnr, cur_psnr)
        if self.max_measure < cur_measure:
            self.max_measure = cur_measure
            self.best_measure_result = u
    
    def get_best_psnr_result(self):
        return self.best_psnr_result
    
    def get_best_ssim_result(self):
        return self.best_ssim_result
    
    def get_best_measure_result(self):
        return self.best_measure_result
   
    def _get_intermediate_results(self, u1, img_H):
        pre_i = torch.clamp(u1 / 255, 0., 1.)
        img_E = util.tensor2uint(pre_i)
        img_H = util.tensor2uint(img_H)
        psnr = util.calculate_psnr(img_E, img_H, border=0)
        ssim = util.calculate_ssim(img_E, img_H, border=0)
        return psnr, ssim

class MixturePnP(nn.Module):

    def init_value(self, f, amf):
        self.ind = ((f != 0) & (f != 255)).long()
        self.im_init = amf * (1 - self.ind) + f * self.ind
        x1 = amf
        x0 = amf
        y1 = f
        y0 = f
        z1 = f
        z0 = f
        gamma1 = torch.zeros_like(f)
        gamma0 = torch.zeros_like(f)
        S1 = torch.zeros_like(f)
        S0 = torch.zeros_like(f)
        W1 = torch.div(20, torch.abs(f - self.im_init))
        # W1 = torch.div(20, torch.abs(f - amf))
        W0 = torch.div(20, torch.abs(f - self.im_init))
        return y1, x1, gamma1, S1, z1, W1, y0, x0, gamma0, S0, z0, W0

    def __init__(self, noise_level, beta, eta, admm_iter_num, denoisor):
        super(MixturePnP, self).__init__()
        
        self.noise_level = noise_level
        self.beta = beta
        self.eta  = eta
        self.admm_iter_num = admm_iter_num

        self.denoisor = denoisor

    def ADMM(self, denoiser, ind, y, sigma, x, gamma, S, z, W, beta, eta):
        S = subproblem_S(y - x, 1./ W**2, gst=False)
        # S = S
        x /= 255
        z = subproblem_z(denoiser, x, gamma, beta, eta, use_drunet=True)
        z *= 255
        x *= 255
        x = subproblem_x(beta, sigma, z, gamma, y, S)
        W = subproblem_W(y, x, S)
        sigma = subproblem_sigma(y, x, S, ind)
        gamma = subproblem_gamma(gamma, beta, x, z)
        return y, sigma, x, gamma, S, z, W, beta, eta
    
    def ADMM_f(self, denoiser, ind, y, sigma, x, gamma, S, z, W, beta, eta):
        S = subproblem_S(y - x, 1./ W**2, gst=False)
        # S = S
        x /= 255
        z = subproblem_z(denoiser, x, gamma, beta, eta, use_drunet=True)
        z *= 255
        x *= 255
        x = subproblem_x(beta, sigma, z, gamma, y, S)
        W = subproblem_W(y, x, S)
        sigma = subproblem_sigma(y, x, S, ind)
        gamma = subproblem_gamma(gamma, beta, x, z)
        return y, sigma, x, gamma, S, z, W, beta, eta
    
    def ADMM_0(self, denoiser, ind, y, sigma, x, gamma, S, z, W, beta, eta):
        S = subproblem_S(y - x, 1./ W**2, gst=False)
        # S = S
        x /= 255
        z = subproblem_z(denoiser, x, gamma, beta, eta, use_drunet=True)
        z *= 255
        x *= 255
        x = subproblem_x(beta, sigma, z, gamma, y, S)
        W = subproblem_W(y, x, S)
        sigma = subproblem_sigma(y, x, S, ind)
        gamma = subproblem_gamma(gamma, beta, x, z)
        return y, sigma, x, gamma, S, z, W, beta, eta

    def forward(self, y, amf, H=None):
        
        checkpoint = Intermediate(H)
        psnr_all = np.zeros(self.admm_iter_num)
        ssim_all = np.zeros(self.admm_iter_num)

        y1, x1, gamma1, S1, z1, W1, y0, x0, gamma0, S0, z0, W0 = self.init_value(y * 255, amf * 255)
        sigma1 = self.noise_level
        beta1 = self.beta
        eta1 = self.eta
        Delta1 = 0
        for k in range(self.admm_iter_num):

            denoisor = self.denoisor.get_denoisor(k)

            if k >= 1:
                y1 = x1 + 0.5 * (y1 - x1) * self.ind - 0.2 * (self.im_init -  x1)*(1 - self.ind)
                y0 = y1

            # out_S1_np = S1.detach().clone().cpu().numpy()[0,0].astype(np.uint8)
            # save_path_S1 = 'inter_results/S1_%d.png'%(k+1)
            # imsave(save_path_S1, out_S1_np)

            out_x1_np = x1.detach().clone().cpu().numpy()[0,0].astype(np.uint8)
            save_path_x1 = 'inter_results/x1_%d.png'%(k+1)
            imsave(save_path_x1, out_x1_np)

            out_y1_np = y1.detach().clone().cpu().numpy()[0,0].astype(np.uint8)
            save_path_y1 = 'inter_results/y1_%d.png'%(k+1)
            imsave(save_path_y1, out_y1_np)

            if k == 0:
                y2, sigma2, x2, gamma2, S2, z2, W2, beta2, eta2 = self.ADMM_0(
                    denoisor, self.ind, y1, sigma1, x1, gamma1, S1, z1, W1, beta1, eta1)
            elif k <= 5:
                y2, sigma2, x2, gamma2, S2, z2, W2, beta2, eta2 = self.ADMM(
                    denoisor, self.ind, y1, sigma1, x1, gamma1, S1, z1, W1, beta1, eta1)
            else:
                y2, sigma2, x2, gamma2, S2, z2, W2, beta2, eta2 = self.ADMM_f(
                    denoisor, self.ind, y1, sigma1, x1, gamma1, S1, z1, W1, beta1, eta1)
                
            out_S2_np = S2.detach().clone().cpu().numpy()[0,0].astype(np.uint8)
            save_path_S2 = 'inter_results/S2_%d.png'%(k+1)
            # imsave(save_path_S2, out_S2_np)

            out_z2_np = z2.detach().clone().cpu().numpy()[0,0].astype(np.uint8)
            save_path_z2 = 'inter_results/z2_%d.png'%(k+1)
            imsave(save_path_z2, out_z2_np)

            Delta2 = (torch.norm(x2-x1) + torch.norm(z2-z1)) / torch.sqrt(torch.tensor(x2.numel()))
            if Delta2 >= 0.5 * Delta1:
                beta2 = min(5 * beta2, 1e15)
            Delta1 = Delta2

            if checkpoint.is_available():
                checkpoint.update(x2)

            # y0, sigma0, x0, gamma0, S0, z0, W0, beta0, eta0 = y1, sigma1, x1, gamma1, S1, z1, W1, beta1, eta1
            y1, sigma1, x1, gamma1, S1, z1, W1, beta1, eta1 = y2, sigma2, x2, gamma2, S2, z2, W2, beta2, eta2

            out_x2_np = x2.detach().clone().cpu().numpy()[0,0].astype(np.uint8)
            save_path_x2 = 'inter_results/x2_%d.png'%(k+1)
            imsave(save_path_x2, out_x2_np)

            # psnr0, ssim0 = get_psnr_ssim(x2,H)
            psnr0, ssim0 = get_psnr_ssim(z2,H)
            psnr_all[k] = psnr0
            ssim_all[k] = ssim0

            # print('idx=%d,psnr=%.2f,ssim=%.2f'%(k,psnr0,ssim0))            
            

        if checkpoint.is_available():
            x2 = checkpoint.get_best_measure_result()

        out_x2_f = x2.detach().clone().cpu().numpy()[0,0].astype(np.uint8)
        save_path_x2 = 'inter_results/xf_%d.png'%(k+1)
        imsave(save_path_x2, out_x2_f)


        return x2  / 255., psnr_all, ssim_all
    
class MixturePnP_q(nn.Module):

    def init_value(self, f, amf):
        self.ind = ((f != 0) & (f != 255)).long()
        self.im_init = amf * (1 - self.ind) + f * self.ind
        x1 = amf
        x0 = amf
        y1 = f
        y0 = f
        z1 = f
        z0 = f
        gamma1 = torch.zeros_like(f)
        gamma0 = torch.zeros_like(f)
        S1 = torch.zeros_like(f)
        S0 = torch.zeros_like(f)
        W1 = torch.div(20, torch.abs(f - self.im_init))
        # W1 = torch.div(20, torch.abs(f - amf))
        W0 = torch.div(20, torch.abs(f - self.im_init))
        return y1, x1, gamma1, S1, z1, W1, y0, x0, gamma0, S0, z0, W0

    def __init__(self, noise_level, beta, eta, admm_iter_num, denoisor, q0):
        super(MixturePnP_q, self).__init__()
        
        self.noise_level = noise_level
        self.beta = beta
        self.eta  = eta
        self.admm_iter_num = admm_iter_num

        self.denoisor = denoisor
        self.q = q0

    def ADMM(self, denoiser, ind, y, sigma, x, gamma, S, z, W, beta, eta):
        S = subproblem_S(y - x, 1./ W**2, gst=True, q=self.q)
        # S = S
        x /= 255
        z = subproblem_z(denoiser, x, gamma, beta, eta, use_drunet=True)
        z *= 255
        x *= 255
        x = subproblem_x(beta, sigma, z, gamma, y, S)
        W = subproblem_W(y, x, S)
        sigma = subproblem_sigma(y, x, S, ind)
        gamma = subproblem_gamma(gamma, beta, x, z)
        return y, sigma, x, gamma, S, z, W, beta, eta
    
    def ADMM_f(self, denoiser, ind, y, sigma, x, gamma, S, z, W, beta, eta):
        S = subproblem_S(y - x, 1./ W**2, gst=True, q=self.q)
        # S = S
        x /= 255
        z = subproblem_z(denoiser, x, gamma, beta, eta, use_drunet=True)
        z *= 255
        x *= 255
        x = subproblem_x(beta, sigma, z, gamma, y, S)
        W = subproblem_W(y, x, S)
        sigma = subproblem_sigma(y, x, S, ind)
        gamma = subproblem_gamma(gamma, beta, x, z)
        return y, sigma, x, gamma, S, z, W, beta, eta
    
    def ADMM_0(self, denoiser, ind, y, sigma, x, gamma, S, z, W, beta, eta):
        S = subproblem_S(y - x, 1./ W**2, gst=True, q=self.q)
        # S = S
        x /= 255
        z = subproblem_z(denoiser, x, gamma, beta, eta, use_drunet=True)
        z *= 255
        x *= 255
        x = subproblem_x(beta, sigma, z, gamma, y, S)
        W = subproblem_W(y, x, S)
        sigma = subproblem_sigma(y, x, S, ind)
        gamma = subproblem_gamma(gamma, beta, x, z)
        return y, sigma, x, gamma, S, z, W, beta, eta

    def forward(self, y, amf, H=None):
        
        checkpoint = Intermediate(H)
        psnr_all = np.zeros(self.admm_iter_num)
        ssim_all = np.zeros(self.admm_iter_num)

        y1, x1, gamma1, S1, z1, W1, y0, x0, gamma0, S0, z0, W0 = self.init_value(y * 255, amf * 255)
        sigma1 = self.noise_level
        beta1 = self.beta
        eta1 = self.eta
        Delta1 = 0
        for k in range(self.admm_iter_num):

            denoisor = self.denoisor.get_denoisor(k)

            if k >= 1:
                y1 = x1 + 0.5 * (y1 - x1) * self.ind - 0.2 * (self.im_init -  x1)*(1 - self.ind)
                y0 = y1

            # out_S1_np = S1.detach().clone().cpu().numpy()[0,0].astype(np.uint8)
            # save_path_S1 = 'inter_results/S1_%d.png'%(k+1)
            # imsave(save_path_S1, out_S1_np)

            out_x1_np = x1.detach().clone().cpu().numpy()[0,0].astype(np.uint8)
            save_path_x1 = 'inter_results/x1_%d.png'%(k+1)
            imsave(save_path_x1, out_x1_np)

            out_y1_np = y1.detach().clone().cpu().numpy()[0,0].astype(np.uint8)
            save_path_y1 = 'inter_results/y1_%d.png'%(k+1)
            imsave(save_path_y1, out_y1_np)

            if k == 0:
                y2, sigma2, x2, gamma2, S2, z2, W2, beta2, eta2 = self.ADMM_0(
                    denoisor, self.ind, y1, sigma1, x1, gamma1, S1, z1, W1, beta1, eta1)
            elif k <= 5:
                y2, sigma2, x2, gamma2, S2, z2, W2, beta2, eta2 = self.ADMM(
                    denoisor, self.ind, y1, sigma1, x1, gamma1, S1, z1, W1, beta1, eta1)
            else:
                y2, sigma2, x2, gamma2, S2, z2, W2, beta2, eta2 = self.ADMM_f(
                    denoisor, self.ind, y1, sigma1, x1, gamma1, S1, z1, W1, beta1, eta1)
                
            out_S2_np = S2.detach().clone().cpu().numpy()[0,0].astype(np.uint8)
            save_path_S2 = 'inter_results/S2_%d.png'%(k+1)
            # imsave(save_path_S2, out_S2_np)

            out_z2_np = z2.detach().clone().cpu().numpy()[0,0].astype(np.uint8)
            save_path_z2 = 'inter_results/z2_%d.png'%(k+1)
            imsave(save_path_z2, out_z2_np)

            Delta2 = (torch.norm(x2-x1) + torch.norm(z2-z1)) / torch.sqrt(torch.tensor(x2.numel()))
            if Delta2 >= 0.5 * Delta1:
                beta2 = min(5 * beta2, 1e15)
            Delta1 = Delta2

            if checkpoint.is_available():
                checkpoint.update(x2)

            # y0, sigma0, x0, gamma0, S0, z0, W0, beta0, eta0 = y1, sigma1, x1, gamma1, S1, z1, W1, beta1, eta1
            y1, sigma1, x1, gamma1, S1, z1, W1, beta1, eta1 = y2, sigma2, x2, gamma2, S2, z2, W2, beta2, eta2

            out_x2_np = x2.detach().clone().cpu().numpy()[0,0].astype(np.uint8)
            save_path_x2 = 'inter_results/x2_%d.png'%(k+1)
            imsave(save_path_x2, out_x2_np)

            # psnr0, ssim0 = get_psnr_ssim(x2,H)
            psnr0, ssim0 = get_psnr_ssim(z2,H)
            psnr_all[k] = psnr0
            ssim_all[k] = ssim0

            # print('idx=%d,psnr=%.2f,ssim=%.2f'%(k,psnr0,ssim0))            
            print("z2:",z2.shape)

        if checkpoint.is_available():
            x2 = checkpoint.get_best_measure_result()

        out_x2_f = x2.detach().clone().cpu().numpy()[0,0].astype(np.uint8)
        save_path_x2 = 'inter_results/xf_%d.png'%(k+1)
        imsave(save_path_x2, out_x2_f)


        return x2  / 255., psnr_all, ssim_all, self.q