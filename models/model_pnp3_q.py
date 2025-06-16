from models.model_plain import ModelPlain
import torch
from collections import OrderedDict
class ModelPnP3_q(ModelPlain):
    """Only test.
    self.L: noise
    self.C: preprocess
    self.H: used to get checkpoint
    """

    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        self.C = data['C'].to(self.device)
        self.H = data['H'].to(self.device)
 
        # self.C = data['C'].to(self.device)
    # ----------------------------------------
    # feed (L, C, H) to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        # print(self.L.shape)
        # print(self.H.shape)
        # print(self.C.shape)
        self.E, psnr_all, ssim_all, q= self.netG(self.L, self.C, self.H)
        return psnr_all, ssim_all, q

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            psnr_all, ssim_all, q = self.netG_forward()
        # self.netG.train()
        return psnr_all, ssim_all, q

    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        # out_dict['psnr_all'] = self.psnr_all
        # out_dict['ssim_all'] = self.ssim_all
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict