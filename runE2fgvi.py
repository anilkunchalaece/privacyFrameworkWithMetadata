# Copied from https://github.com/anilkunchalaece/E2FGVI.git
# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import importlib
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import torch

from lib.e2fgvi.core.utils import to_tensors

parser = argparse.ArgumentParser(description="E2FGVI")
parser.add_argument("-v", "--video", type=str, required=True)
parser.add_argument("-c", "--ckpt", type=str, default="data/e2fgvi/E2FGVI-HQ-CVPR22.pth")
parser.add_argument("-m", "--mask", type=str, required=True)
parser.add_argument("-o", "--output_dir", type=str, required=True)
parser.add_argument("--model", type=str, default="e2fgvi_hq")
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--num_ref", type=int, default=-1)
parser.add_argument("--neighbor_stride", type=int, default=5)
parser.add_argument("--savefps", type=int, default=24)

# args for e2fgvi_hq (which can handle videos with arbitrary resolution)
parser.add_argument("--set_size", action='store_true', default=False)
parser.add_argument("--width", type=int)
parser.add_argument("--height", type=int)

args = parser.parse_args()

ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps


# sample reference frames from the whole video
def get_ref_index(f, neighbor_ids, length,ref_length=args.step):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref // 2))
        end_idx = min(length, f + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index


# read frame-wise masks
def read_mask(mpath, size):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for mp in mnames:
        m = Image.open(os.path.join(mpath, mp))
        m = m.resize(size, Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m,
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                       iterations=4)
        masks.append(Image.fromarray(m * 255))
    return masks

def get_mask_tensors(mask_names,size):
    masks = []

    for mp in mask_names:
        m = Image.open(mp)
        # m = m.resize(size[:2], Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m,
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                       iterations=4)
        masks.append(Image.fromarray(m * 255))
    masks = to_tensors()(masks).unsqueeze(0)
    return masks    

# get mask filenames 
def get_mask_file_names(mpath) :
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    mnames = [os.path.join(mpath,f) for f in mnames]
    return mnames    


#  read frames from video
def read_frame_from_videos(args):
    vname = args.video
    frames = []
    if args.use_mp4:
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        count = 0
        while success:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image)
            success, image = vidcap.read()
            count += 1
    else:
        lst = os.listdir(vname)
        lst.sort()
        fr_lst = [vname + '/' + name for name in lst]
        for fr in fr_lst:
            image = cv2.imread(fr)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image)
    return frames

def get_image_tensors(img_file_names) :
    frames = []
    for f in img_file_names :
        image = cv2.imread(f)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image)
    frames = to_tensors()(frames).unsqueeze(0) * 2 - 1
    return frames        


# get frame file names
def get_frame_file_names(args):
    vname = args.video
    frames = []
    lst = os.listdir(vname)
    lst.sort()
    fr_lst = [os.path.join(vname,name) for name in lst]
    return fr_lst


# resize frames
def resize_frames(frames, size=None):
    if size is not None:
        frames = [f.resize(size) for f in frames]
    else:
        size = frames[0].size
    return frames, size


def main_worker():
    # set up models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "e2fgvi":
        size = (432, 240)
    elif args.set_size:
        size = (args.width, args.height)
    else:
        size = None

    net = importlib.import_module('lib.e2fgvi.model.' + args.model)
    model = net.InpaintGenerator().to(device)
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data)
    print(f'Loading model from: {args.ckpt}')
    model.eval()

    # prepare datset
    args.use_mp4 = True if args.video.endswith('.mp4') else False
    print(
        f'Loading videos and masks from: {args.video} | INPUT MP4 format: {args.use_mp4}'
    )
    frames = get_frame_file_names(args)
    # frames = read_frame_from_videos(args)
    # frames, size = resize_frames(frames, size)
    size = cv2.imread(frames[0]).shape
    h, w = size[0], size[1]
    video_length = len(frames)

    # imgs = to_tensors()(frames).unsqueeze(0) * 2 - 1
    # frames = [np.array(f).astype(np.uint8) for f in frames]

    # masks = read_mask(args.mask, size)
    # binary_masks = [
    #     np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks
    # ]
    # masks = to_tensors()(masks).unsqueeze(0)
    # imgs, masks = imgs.to(device), masks.to(device)
    masks = get_mask_file_names(args.mask)

    # completing holes by e2fgvi
    print(f'Start test...')

    for f in tqdm(range(0, video_length, neighbor_stride)):
        
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                            min(video_length, f + neighbor_stride + 1))
        ]

        ref_ids = get_ref_index(f, neighbor_ids, video_length)
        
        ref_diff = [abs(x-f) for x in ref_ids] # get the closet value in ref_ids
        ref_diff_index = ref_diff.index(min(ref_diff))
        
        # take the references around the current frame
        if ref_diff_index < 3 :
            ref_start_idx = 0
            ref_end_idx = 6
        elif ref_diff_index > video_length - 3 :
            ref_start_idx = video_length - 6
            ref_end_idx = video_length 
        else :
            ref_start_idx = ref_diff_index - 3
            ref_end_idx = ref_diff_index + 3

        #consider only first 20 images as references, to avoid memory errors
        ref_ids = ref_ids[ref_start_idx:ref_end_idx] # What is the effect of performance ? TODO

        # selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
        # selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
        
        selected_ids = neighbor_ids+ref_ids
        selected_img_names = [frames[idx] for idx in selected_ids]
        selected_mask_names = [masks[idx] for idx in selected_ids]

        # print(F"selecteid => {len(selected_ids)}, ref_ids {len(ref_ids)}")

        selected_imgs = get_image_tensors(selected_img_names).to(device)
        selected_masks = get_mask_tensors(selected_mask_names, size).to(device)

        with torch.no_grad():
            masked_imgs = selected_imgs * (1 - selected_masks)
            mod_size_h = 60
            mod_size_w = 108
            h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
            w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [3])],
                3)[:, :, :, :h + h_pad, :]
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [4])],
                4)[:, :, :, :, :w + w_pad]
            pred_imgs, _ = model(masked_imgs, len(neighbor_ids))
            pred_imgs = pred_imgs[:, :, :h, :w]
            pred_imgs = (pred_imgs + 1) / 2
            pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255 # original

            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                c_frame = np.array(Image.fromarray(cv2.cvtColor(cv2.imread(frames[idx]), cv2.COLOR_BGR2RGB))).astype(np.uint8)
                c_binary_mask =  np.expand_dims((np.array(cv2.imread(masks[idx])) != 0).astype(np.uint8), 2).squeeze()

                # img = np.array(pred_imgs[i]).astype(
                #     np.uint8) * binary_masks[idx] + frames[idx] * (
                #         1 - binary_masks[idx])
                img = np.array(pred_imgs[i]).astype(
                    np.uint8) * c_binary_mask + c_frame * (
                        1 - c_binary_mask)                
                # if comp_frames[idx] is None:
                #     comp_frames[idx] = img
                # else:
                #     comp_frames[idx] = comp_frames[idx].astype(
                #         np.float32) * 0.5 + img.astype(np.float32) * 0.5

                fToSave = os.path.join(args.output_dir, F"{idx:06d}.{frames[0].split('.')[-1]}")
                if not os.path.isfile(fToSave) :
                    img_to_save = img
                else :
                    img_to_save = cv2.cvtColor(cv2.imread(fToSave),cv2.COLOR_RGB2BGR).astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                cv2.imwrite(fToSave,cv2.cvtColor(img_to_save,cv2.COLOR_BGR2RGB))


if __name__ == '__main__':
    main_worker()