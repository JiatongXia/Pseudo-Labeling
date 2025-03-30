import argparse
import os
import lpips

import torch
from imageio import imread
from tqdm import tqdm
import numpy as np


import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

def color_map_forward(rgb):
    return rgb.astype(np.float32) / 255

class Evaluator:
    def __init__(self):
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda().eval()
        # self.loss_fn_alex = lpips.LPIPS(net='alex').cuda().eval()

    def eval_metrics_img(self,gt_img, pr_img):
        gt_img = color_map_forward(gt_img)
        pr_img = color_map_forward(pr_img)
        psnr = tf.image.psnr(tf.convert_to_tensor(gt_img), tf.convert_to_tensor(pr_img), 1.0, )
        ssim = tf.image.ssim(tf.convert_to_tensor(gt_img), tf.convert_to_tensor(pr_img), 1.0, )
        with torch.no_grad():
            gt_img_th = torch.from_numpy(gt_img).cuda().permute(2,0,1).unsqueeze(0) * 2 - 1
            pr_img_th = torch.from_numpy(pr_img).cuda().permute(2,0,1).unsqueeze(0) * 2 - 1
            score = float(self.loss_fn_vgg(gt_img_th, pr_img_th).flatten()[0].cpu().numpy())
        return float(psnr), float(ssim), score


    def eval(self, dir_gt, dir_pr):
        results=[]
        num = len(os.listdir(dir_pr))
        gt_imgs = sorted([os.path.join(dir_gt, gts) for gts in os.listdir(dir_gt)])
        pr_imgs = sorted([os.path.join(dir_pr, prs) for prs in os.listdir(dir_pr)])
        for k in tqdm(range(0, num)):
            pr_img = imread(pr_imgs[k])
            gt_img = imread(gt_imgs[k])

            psnr, ssim, lpips_score = self.eval_metrics_img(gt_img, pr_img)
            results.append([psnr,ssim,lpips_score])
            
        results = np.asarray(results)
        print('psnr : ', results[:,0])
        print('ssim : ', results[:,1])
        print('lpips : ', results[:,2])
        psnr, ssim, lpips_score = np.mean(results,0)

        msg=f'psnr {psnr:.4f} ssim {ssim:.4f} lpips {lpips_score:.4f}'
        print(msg)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_gt', type=str)
    parser.add_argument('--dir_pr', type=str)
    flags = parser.parse_args()
    evaluator = Evaluator()
    evaluator.eval(flags.dir_gt, flags.dir_pr)
