import numpy as np
import os
import SimpleITK as sitk
import re
import cv2
# import hausdorff
# import seg_metrics.seg_metrics as sg

def file_name(file_dir):
    L = []
    path_list = os.listdir(file_dir)
    path_list.sort()  # 对读取的路径进行排序
    for filename in path_list:
        if 'nii' in filename:
            L.append(os.path.join(filename))
    return L


def computeQualityMeasures(lP, lT):
    quality = dict()
    #labelPred = sitk.GetImageFromArray(lP)
    #labelTrue = sitk.GetImageFromArray(lT)
    labelPred = lP
    labelTrue = lT
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue, labelPred)
    quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()

    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue, labelPred)
    quality["dice"] = dicecomputer.GetDiceCoefficient()

    return quality


UNet1path = '/data/zk/1/data/test_full/UNet-1/'
UNet2path = '/data/zk/1/data/test_full/UNet-2/'
UNet3path = '/data/zk/1/data/test_full/UNet-3/'
UNet4path = '/data/zk/1/data/test_full/UNet-4/'
imgpath = '/data/zk/1/data/test/'
savepath = '/data/zk/1/data/test_fusion/'
# maskpath = '/data/zk/1/data/masks/masks/'
# maskNpath = '/data/zk/1/data/masks/TMJ_masks/'

imgnames = file_name(imgpath)
# masknames = file_name(maskpath)
# prednames = file_name(predpath)

# labels_num = np.zeros(len(prednames))
NUM = []
P = []
s1 = 0
s2 = 0
for j in range(len(imgnames)):
    i = j+0
    UNet1 = sitk.ReadImage(UNet1path + imgnames[i]+'.gz')
    pil_img1 = sitk.GetArrayFromImage(UNet1)
    UNet2 = sitk.ReadImage(UNet2path + imgnames[i]+'.gz')
    pil_img2 = sitk.GetArrayFromImage(UNet2)
    UNet3 = sitk.ReadImage(UNet3path + imgnames[i]+'.gz')
    pil_img3 = sitk.GetArrayFromImage(UNet3)
    UNet4 = sitk.ReadImage(UNet4path + imgnames[i]+'.gz')
    pil_img4 = sitk.GetArrayFromImage(UNet4)
    image = sitk.ReadImage(imgpath + imgnames[i])
    pil_img5 = sitk.GetArrayFromImage(image)
    pil_img1 = pil_img1 / ((pil_img1.max() - pil_img1.min()) * 0.5)
    pil_img2 = (pil_img2 / ((pil_img2.max() - pil_img2.min()) * 0.5))
    pil_img3 = pil_img3 / ((pil_img3.max() - pil_img3.min()) * 0.5)
    pil_img4 = pil_img4 / ((pil_img4.max() - pil_img4.min()) * 0.5)
    pil_img5 = (pil_img5 / ((pil_img5.max() - pil_img5.min()) * 0.5)).astype(np.float32)
    pil_img = np.vstack((np.expand_dims(pil_img1, axis=0), np.expand_dims(pil_img2, axis=0),
                         np.expand_dims(pil_img3, axis=0), np.expand_dims(pil_img4, axis=0),
                         np.expand_dims(pil_img5, axis=0)))
    pil_img = pil_img.transpose((1, 2, 3, 0))
    gt = sitk.GetImageFromArray(pil_img)
    sitk.WriteImage(gt, savepath+imgnames[i]+'.gz')


    # gt = sitk.GetArrayFromImage(gt)
    # pred = sitk.GetArrayFromImage(pred)
    # gt = (gt/2).astype(np.uint8)
    # gt = sitk.GetImageFromArray(gt)
    # sitk.WriteImage(gt, './data/1gt.nii.gz')
    # gt = sitk.GetImageFromArray(gt.astype(np.uint8))
    # pred = sitk.GetImageFromArray(pred.astype(np.uint8))
    # gt = sitk.GetArrayFromImage(gt)
    # sitk.WriteImage(img, imgNpath+'TMJ_'+f'{i}'+'.nii.gz')
    # sitk.WriteImage(mask, maskNpath + 'TMJ_' + f'{i}' + '.nii.gz')
    # a=re.findall("\d+", gtnames[i])
    # sitk.WriteImage(gt, gtpath+re.findall("\d+", gtnames[i])[0] + '.nii.gz')
    # quality = computeQualityMeasures(pred, gt)
    # s1 = s1 + quality['dice']
    # s2 = s2 + quality['Hausdorff']
    # print(quality)
    # if i == (len(gtnames)-1):
    #   print('average_dice:', s1/len(gtnames))
    #   print('average_Hausdorff:', s2/len(gtnames))
