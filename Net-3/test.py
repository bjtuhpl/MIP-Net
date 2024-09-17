import SimpleITK as sitk
import numpy as np
import cv2
import scipy

def convolve3d(img,kernel,mode='same'):
    for i in np.arange(img.shape[0]):
        for j in np.arange(img.shape[1]):
            oneline = img[i,j,:]
            img[i,j,:] = np.convolve(oneline, kernel[0], mode='same')
    for i in np.arange(img.shape[1]):
        for j in np.arange(img.shape[2]):
            oneline = img[:,i,j]
            img[:,i,j] = np.convolve(oneline, kernel[1], mode='same')
    for i in np.arange(img.shape[0]):
        for j in np.arange(img.shape[2]):
            oneline = img[i,:,j]
            img[i,:,j]=np.convolve(oneline,kernel[2],mode='same')
    return  img

img=sitk.ReadImage('/data/zk/1/data/imgs/TMJ_imgs/TMJ_1.nii.gz')
img=sitk.GetArrayFromImage(img)
v=np.array([[1,1,1],[1,1,1],[1,1,1]])
temp=img
out=cv2.GaussianBlur(img,(15,15),0)
outImg=np.ones((240,240,240))
for i in range(0,240):
    for j in range(0,240):
        for k in range(0,240):
            outImg[i][j][k]=out[i*2][j*2][k*2];
# out=convolve3d(temp,kernel=v,mode='same')
outImg=outImg.astype(np.int16)
out=sitk.GetImageFromArray(outImg)
sitk.WriteImage(out, '/data/zk/gp/r1.nii.gz')




