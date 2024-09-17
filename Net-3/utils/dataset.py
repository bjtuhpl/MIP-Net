from os.path import splitext, split
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import SimpleITK as sitk
from skimage import transform
import random
# import cv2


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        a = list(np.linspace(0, 0, num=1))
        # a=random.sample(range(0,71),42)
        self.ids = [str(int(i)) for i in a]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, pil_img0, scale):
        w, h, d = pil_img.shape
        newW, newH, newD = int(scale * w), int(scale * h), int(scale * d)
        assert newW > 0 and newH > 0, 'Scale is too small'
        # if pil_img.max()==2:
        #     pil_img=(pil_img*127.5).astype(np.uint8)
        #     pil_img = transform.resize(pil_img, (newW, newH, newD), anti_aliasing=False)
        #     #ret1, pil_img = cv2.threshold(pil_img, 0.7, 1, cv2.THRESH_BINARY)
        if pil_img.max() == 1:
            pil_img = (pil_img * 255).astype(np.uint8)
            pil_img = transform.resize(pil_img, (newW, newH, newD), anti_aliasing=False)
            # ret1, pil_img = cv2.threshold(pil_img, 0.7, 1, cv2.THRESH_BINARY)
        elif pil_img0 is None:
            mn = pil_img.mean()
            std = pil_img.std()
            pil_img = (pil_img - mn) / std
        else:
            mn = pil_img.mean()
            std = pil_img.std()
            pil_img = (pil_img - mn) / std
            mn = pil_img0.mean()
            std = pil_img0.std()
            pil_img0 = (pil_img0 - mn) / std
            pil_img0 = transform.resize(pil_img0, (int(newW), newH, newD), anti_aliasing=False)
            pil_img = transform.resize(pil_img, (int(newW), newH, newD), anti_aliasing=False)
            pil_img = np.vstack((np.expand_dims(pil_img, axis=0), np.expand_dims(pil_img0, axis=0)))
       


        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 3:
            img_nd = np.expand_dims(img_nd, axis=0)

        # HWC to CHW
        # img_trans = img_nd.transpose((3, 0, 1, 2))
        # if img_trans.max() > 500:
        #     img_trans = img_trans / 32768
        # if img_trans.max() > 2:
        #     img_trans = img_trans / 255

        return img_nd

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir +'TMJ_'+ idx + '.nii.gz')
        img_file = glob(self.imgs_dir + 'TMJ_'+idx + '.nii.gz')
        img0_file=glob('/home/zhangkai/zk/results/TMJ_IPG/stage2/train_72Img_res/'+'TMJ_'+idx+'.nii.gz')
        img1_file = glob('/home/zhangkai/zk/data/stage3/stage1/train1_index/' + 'TMJ_' + idx + '.nii.gz')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        # mask = Image.open(mask_file[0])
        mask = sitk.ReadImage(mask_file[0])
        mask = sitk.GetArrayFromImage(mask)
        img = sitk.ReadImage(img_file[0])
        img = sitk.GetArrayFromImage(img)
        img0=sitk.ReadImage(img0_file[0])
        img0=sitk.GetArrayFromImage(img0)
        img1=sitk.ReadImage(img1_file[0])
        img1=sitk.GetArrayFromImage(img1)


        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img,img0, self.scale)
        mask = self.preprocess(mask,img0, self.scale)
        img0=self.preprocess(img0,None,1)
        img1 = self.preprocess(img1, None, 1)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask), 'img0': torch.from_numpy(img0), 'img1': torch.from_numpy(img1)}
