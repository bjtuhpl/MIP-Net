import numpy as np
import os
import SimpleITK as sitk
import nibabel as nib
import surface_distance as surfdist
# import cv2
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


def computeQualityMeasures(mask_pred, mask_gt, spacing_mm_s):

    quality = dict()
    # #labelPred = sitk.GetImageFromArray(lP)
    # #labelTrue = sitk.GetImageFromArray(lT)
    # labelPred = lP
    # labelTrue = lT
    # hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    # hausdorffcomputer.Execute(labelTrue, labelPred)
    # quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
    # quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()
    #
    # dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    # dicecomputer.Execute(labelTrue, labelPred)
    # quality["dice"] = dicecomputer.GetDiceCoefficient()
    gt = mask_gt.dataobj[:,:,:]
    pred = mask_pred.dataobj[:,:,:]

    gt = gt.astype(np.bool)
    pred = pred.astype(np.bool)

    surface_distances = surfdist.compute_surface_distances(gt, pred, spacing_mm=spacing_mm_s)
    hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)
    hd_dist = surfdist.compute_robust_hausdorff(surface_distances, 100)
    surf_dist = surfdist.compute_average_surface_distance(surface_distances)
    avg_surf_dist = (surf_dist[0] + surf_dist[1]) / 2
    quality['hd95'] = hd_dist_95
    quality['hd'] = hd_dist
    quality['asd'] = avg_surf_dist
    mH.append(hd_dist_95)

    return quality


gtpath = './data/41Img/recover/'
predpath = './data/41Img/gt/'

gtnames = file_name(gtpath)
prednames = file_name(predpath)

labels_num = np.zeros(len(prednames))
NUM = []
spacing_mm =(0.125,0.125,0.125)
mH = []
P = []
s1 = 0
s2 = 0
s3 = 0
for i in range(len(gtnames)):
    gt = nib.load(gtpath + gtnames[i])
    pred = nib.load(predpath + gtnames[i])
    # pred = sitk.GetArrayFromImage(pred)
    # gt = sitk.GetArrayFromImage(gt)
    # pred = sitk.GetArrayFromImage(pred)
    # gt = (gt/2).astype(np.uint8)
    # gt = sitk.GetImageFromArray(gt)
    # sitk.WriteImage(gt, './data/1gt.nii.gz')
    # gt = sitk.GetImageFromArray(gt.astype(np.uint8))
    # sitk.WriteImage(gt, './data/test/gt/95.nii.gz')

    quality = computeQualityMeasures(pred, gt, spacing_mm)
    s1 = s1 + quality['hd']
    s2 = s2 + quality['hd95']
    s3 = s3 + quality['asd']
    print(quality)
    if i == (len(gtnames)-1):
      print('average_dice:', s1/len(gtnames))
      print('average_Hausdorff_95:', s2/len(gtnames))
      print('average_asd:', s3/len(gtnames))
