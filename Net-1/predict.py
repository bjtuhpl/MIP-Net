import argparse
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
# from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
import SimpleITK as sitk
from dice_loss import dice_coeff
from skimage import transform

def predict_img(net,
                full_img,
                device,
                scale_factor=0.267,
                out_threshold=0.6):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    img_pro = (img).squeeze(0).numpy()
    img_show = sitk.GetImageFromArray(img_pro)
    #sitk.WriteImage(img_show, './data/pred/scale0.2_002input.nii')
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        max_out = output.cpu().numpy().max()

        # if net.n_classes > 1:
        #     probs = F.softmax(output, dim=1)
        # else:
        #     probs = torch.sigmoid(output)
        #     max0 = probs.cpu().numpy().max()

        probs = output.squeeze()
        #
        # tf = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         #transforms.Resize((481,481,481)),
        #         transforms.ToTensor()
        #     ]
        # )

        #probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
    return full_mask
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='checkpoints/scale0.25_72Img_dice/CP_epoch300.pth',
                        metavar='FILE',help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', default='/home/zhangkai/zk/data/test_img/', metavar='INPUT', nargs='+',
                        help='filenames of input images')

    parser.add_argument('--output', '-o', default='/home/zhangkai/zk/results/TMJ_IPG/stage1/test_dice_new/', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.6)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.25)

    parser.add_argument('--gt', '-g', default='data/masks/', metavar='INPUT', nargs='+')

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))

    # elif len(in_files) != len(args.output):
    #     logging.error("Input files and output files are not of the same length")
    #     raise SystemExit()
    else:
        out_files = args.output

    return out_files

	
def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def file_name(file_dir):
    L = []
    path_list = os.listdir(file_dir)
    path_list.sort()
    for filename in path_list:
        if 'nii' in filename:
            L.append((os.path.join(filename)))
    return L

if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)
    #gt_files = args.gt

    #gt = sitk.ReadImage(gt_files[0])
    #gt = sitk.GetArrayFromImage(gt)

    net = UNet(n_channels=1, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(file_name(in_files)):
        logging.info("\nPredicting image {} ...".format(fn))

        img = sitk.ReadImage(in_files + fn)
        img = sitk.GetArrayFromImage(img)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            # out_fn = out_files[i]
            #result = sitk.GetImageFromArray(mask_to_image(mask))    
            #max_mask = mask.max()
            # a = (mask*255).astype(np.uint8)
            ret1, output = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)
            b = transform.resize(output, (481,481,481), anti_aliasing=False, preserve_range=True)
            ret1, b = cv2.threshold(b, 0.5, 1, cv2.THRESH_BINARY)

            # result = sitk.GetImageFromArray(mask)
            result = sitk.GetImageFromArray(b.astype(np.uint8))
            #result.save(out_files[i]   

            sitk.WriteImage(result, out_files + fn +'.gz')
            #a = torch.from_numpy(a/255)
            #gt = torch.from_numpy((gt*255).astype(np.uint8))
            #gt = transform.resize((gt*255).astype(np.uint8), (134, 134, 134), anti_aliasing=False)
            #gt = torch.from_numpy(gt)
            #print(dice_coeff(a.double(), gt).item())
	    

            # logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
