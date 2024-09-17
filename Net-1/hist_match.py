import SimpleITK as sitk
import numpy as np
import glob
from os.path import splitext, split
import matplotlib.pyplot as plt

imgs_dir = '/home/zhangkai/zk/data/TMJ_imgs/'


def histogram_matching():
    # Load the template image
    # template = nib.load(reference_nii)
    img_dir = glob.glob('/home/zhangkai/zk/data/17.nii')
    # img_dir.remove('./data/imgs/1.nii')
    for dir in img_dir:
        template = sitk.ReadImage('/home/zhangkai/zk/data/TMJ_imgs/TMJ_1.nii.gz')
        nt_data = sitk.GetArrayFromImage(template)

        # Load the patient image
        # patient = nib.load(input_nii)
        patient = sitk.ReadImage(dir)
        pt_data = sitk.GetArrayFromImage(patient)

        # Stores the image data shape that will be used later
        oldshape = pt_data.shape

        # Converts the data arrays to single dimension and normalizes by the maximum
        nt_data_array = nt_data.ravel()
        pt_data_array = pt_data.ravel()

        # get the set of unique pixel values and their corresponding indices and counts
        s_values, bin_idx, s_counts = np.unique(pt_data_array, return_inverse=True, return_counts=True)
        # plt.axes(yscale="log")
        # n, bins, patches = plt.hist(s_values, 50, facecolor='blue', alpha=0.5)
        # plt.plot(bins, s_counts, 'r--')

        # plt.hist(nt_data_array, bins=50, rwidth=0.85, alpha=0.7)
        #
        # plt.xlabel('Image Intensity')
        # plt.ylabel('Pixel Count')
        # plt.show()

        t_values, t_counts = np.unique(nt_data_array, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        # Reshapes the corresponding values to the indexes and reshapes the array to input
        out_img = interp_t_values[bin_idx].reshape(oldshape)
        # out_img_arry = out_img.ravel()
        # plt.hist(out_img_arry, bins=50, rwidth=0.85, alpha=0.7)
        #
        # plt.xlabel('Image Intensity')
        # plt.ylabel('Pixel Count')
        # plt.show()

        out_img = sitk.GetImageFromArray(out_img.astype(np.int16))
        sitk.WriteImage(out_img, '/home/zhangkai/zk/data/17_hist.nii.gz')
        # final_image_data[indx] = 0

        # Saves the output data
        # img = nib.Nifti1Image(final_image_data, patient.affine, patient.header)
        # nib.save(img, output_nii)


his_match = histogram_matching()
