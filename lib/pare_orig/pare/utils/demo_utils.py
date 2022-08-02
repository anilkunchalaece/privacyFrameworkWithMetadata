# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import cv2
import time
import json
import torch
import subprocess
import numpy as np
import os.path as osp
from pytube import YouTube
from collections import OrderedDict

from .smooth_bbox import get_smooth_bbox_params, get_all_bbox_params
from .vibe_image_utils import get_single_image_crop_demo
from .geometry import rotation_matrix_to_angle_axis

from pare.models.head.smpl_head import SMPL,SMPLHead
from pare.core import config, constants
from pare.utils import geometry

# from lib.smplify.temporal_smplify import TemporalSMPLify
import shutil

def preprocess_video(video, joints2d, bboxes, frames, scale=1.0, crop_size=224):
    """
    Read video, do normalize and crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.

    :param video (ndarray): input video
    :param joints2d (ndarray, NxJx3): openpose detections
    :param bboxes (ndarray, Nx5): bbox detections
    :param scale (float): bbox crop scaling factor
    :param crop_size (int): crop width and height
    :return: cropped video, cropped and normalized video, modified bboxes, modified joints2d
    """

    if joints2d is not None:
        bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
        bboxes[:,2:] = 150. / bboxes[:,2:]
        bboxes = np.stack([bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,2]]).T

        video = video[time_pt1:time_pt2]
        joints2d = joints2d[time_pt1:time_pt2]
        frames = frames[time_pt1:time_pt2]

    shape = video.shape

    temp_video = np.zeros((shape[0], crop_size, crop_size, shape[-1]))
    norm_video = torch.zeros(shape[0], shape[-1], crop_size, crop_size)

    for idx in range(video.shape[0]):

        img = video[idx]
        bbox = bboxes[idx]

        j2d = joints2d[idx] if joints2d is not None else None

        norm_img, raw_img, kp_2d = get_single_image_crop_demo(
            img,
            bbox,
            kp_2d=j2d,
            scale=scale,
            crop_size=crop_size)

        if joints2d is not None:
            joints2d[idx] = kp_2d

        temp_video[idx] = raw_img
        norm_video[idx] = norm_img

    temp_video = temp_video.astype(np.uint8)

    return temp_video, norm_video, bboxes, joints2d, frames


def customShape(pare_results):
    # Calculate average shape
    shape_list = np.zeros(shape=(len(pare_results.keys()),10))
    for idx, k in enumerate(pare_results.keys()) :
        shape_list[idx,:] = np.average(pare_results[k]['betas'],axis=0)        
    avg_shape = np.average(shape_list,axis=0)

    # avg_shape = ((np.random.rand(1,10) - 0.5)*0.06) # take random shape instead of average shape
    avg_shape = np.array([[0.00495731,-0.00761945,-0.01329031,-0.01045073,0.02202598,0.0265389 ,-0.01466284,-0.01419266,-0.02254305,-0.010054 ]])
    # print(pare_results[1].keys())
    # print(pare_results[k]['betas'])

    # generate vertices with average shape
    smpl = SMPLHead(config.SMPL_MODEL_DIR)
    for idx, k in enumerate(pare_results.keys()):
        # avg_shape_n = np.tile(avg_shape, (pare_results[k]['betas'].shape[0], 1))
        # print(pare_results[k]['pose'].shape)
        # in PARE pose represented rotation matix representation i.e 24x3x3
        # we need to convert it axis angle representation
        # ref - https://github.com/mkocabas/PARE/issues/4
        # out = smpl(betas=avg_shape_n[1,:],body_pose=geometry.batch_rot2aa(torch.from_numpy(pare_results[k]['pose'][1,:,:,:])))
        # print(out.verts.shape)

        avg_shape_n = torch.from_numpy(np.tile(avg_shape, (pare_results[k]['betas'].shape[0], 1)).astype(np.float32))
        body_pose = torch.from_numpy(pare_results[k]['pose'].astype(np.float32))
        # logger.info(body_pose.shape)
        out = smpl(rotmat=body_pose, shape=avg_shape_n)

        pare_results[k]['verts'] = out["smpl_vertices"]

        # for i, f in enumerate(pare_results[k]['pose']) :
        #     betas = torch.from_numpy(avg_shape.astype(np.float32)).reshape(1,-1)
        #     body_pose = geometry.batch_rot2aa(torch.from_numpy(f.astype(np.float32)))[1:,:].reshape(1,-1)
        #     global_orient = geometry.batch_rot2aa(torch.from_numpy(f.astype(np.float32)))[1,:].reshape(1,-1)
        #     trans = torch.from_numpy(pare_results[k]['pred_cam'][i].astype(np.float32))
        #     body_pose = torch.from_numpy(f.astype(np.float32)).reshape(1,24,3,3)

        #     # out = smpl(betas=betas,body_pose=body_pose,global_orient=global_orient,trans=trans)
        #     out = smpl(rotmat=body_pose, shape=betas)
        #     # print(out)
        #     pare_results[k]['verts'][i,:] = torch.squeeze(out["smpl_vertices"])
    return pare_results

def download_youtube_clip(url, download_folder):
    return YouTube(url).streams.first().download(output_path=download_folder)


# def smplify_runner(
#         pred_rotmat,
#         pred_betas,
#         pred_cam,
#         j2d,
#         device,
#         batch_size,
#         lr=1.0,
#         opt_steps=1,
#         use_lbfgs=True,
#         pose2aa=True
# ):
#     smplify = TemporalSMPLify(
#         step_size=lr,
#         batch_size=batch_size,
#         num_iters=opt_steps,
#         focal_length=5000.,
#         use_lbfgs=use_lbfgs,
#         device=device,
#         # max_iter=10,
#     )
#     # Convert predicted rotation matrices to axis-angle
#     if pose2aa:
#         pred_pose = rotation_matrix_to_angle_axis(pred_rotmat.detach()).reshape(batch_size, -1)
#     else:
#         pred_pose = pred_rotmat
#
#     # Calculate camera parameters for smplify
#     pred_cam_t = torch.stack([
#         pred_cam[:, 1], pred_cam[:, 2],
#         2 * 5000 / (224 * pred_cam[:, 0] + 1e-9)
#     ], dim=-1)
#
#     gt_keypoints_2d_orig = j2d
#     # Before running compute reprojection error of the network
#     opt_joint_loss = smplify.get_fitting_loss(
#         pred_pose.detach(), pred_betas.detach(),
#         pred_cam_t.detach(),
#         0.5 * 224 * torch.ones(batch_size, 2, device=device),
#         gt_keypoints_2d_orig).mean(dim=-1)
#
#     best_prediction_id = torch.argmin(opt_joint_loss).item()
#     pred_betas = pred_betas[best_prediction_id].unsqueeze(0)
#     # pred_betas = pred_betas[best_prediction_id:best_prediction_id+2] # .unsqueeze(0)
#     # top5_best_idxs = torch.topk(opt_joint_loss, 5, largest=False)[1]
#     # breakpoint()
#
#     start = time.time()
#     # Run SMPLify optimization initialized from the network prediction
#     # new_opt_vertices, new_opt_joints, \
#     # new_opt_pose, new_opt_betas, \
#     # new_opt_cam_t, \
#     output, new_opt_joint_loss = smplify(
#         pred_pose.detach(), pred_betas.detach(),
#         pred_cam_t.detach(),
#         0.5 * 224 * torch.ones(batch_size, 2, device=device),
#         gt_keypoints_2d_orig,
#     )
#     new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)
#     # smplify_time = time.time() - start
#     # print(f'Smplify time: {smplify_time}')
#     # Will update the dictionary for the examples where the new loss is less than the current one
#     update = (new_opt_joint_loss < opt_joint_loss)
#
#     new_opt_vertices = output['verts']
#     new_opt_cam_t = output['theta'][:,:3]
#     new_opt_pose = output['theta'][:,3:75]
#     new_opt_betas = output['theta'][:,75:]
#     new_opt_joints3d = output['kp_3d']
#
#     return_val = [
#         update, new_opt_vertices.cpu(), new_opt_cam_t.cpu(),
#         new_opt_pose.cpu(), new_opt_betas.cpu(), new_opt_joints3d.cpu(),
#         new_opt_joint_loss, opt_joint_loss,
#     ]
#
#     return return_val



def trim_videos(filename, start_time, end_time, output_filename):
    command = ['ffmpeg',
               '-i', '"%s"' % filename,
               '-ss', str(start_time),
               '-t', str(end_time - start_time),
               '-c:v', 'libx264', '-c:a', 'copy',
               '-threads', '1',
               '-loglevel', 'panic',
               '"%s"' % output_filename]
    # command = ' '.join(command)
    subprocess.call(command)


def video_to_images(vid_file, img_folder=None, return_info=False):
    if img_folder is None:
        img_folder = osp.join('/tmp', osp.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(osp.join(img_folder, '000001.png')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder

def renameSrcDir(srcDir) :
    # Check fileNames -> files should start with 0, if not rename them
    fList = sorted(os.listdir(srcDir), key= lambda f: int(f.split(".")[0]))
    startIdx = int(fList[0].split(".")[0])
    endIdx = int(fList[-1].split(".")[0])
    # print(F"hi there {startIdx}")
    # if startIdx != 0 :
        # print("startIdx not zero")
    for idx, f in enumerate(fList) :
        srcName = os.path.join(srcDir,f)
        # desName = os.path.join(srcDir,F"{int(f.split('.')[0]) - startIdx:06d}.{f.split('.')[-1]}")
        desName = os.path.join(srcDir,F"{idx:06d}.{fList[0].split('.')[1]}")
        # print(srcName, desName)
        shutil.move(srcName,desName)    


def download_url(url, outdir):
    print(f'Downloading files from {url}')
    cmd = ['wget', '-c', url, '-P', outdir]
    subprocess.call(cmd)


def download_ckpt(outdir='data/vibe_data', use_3dpw=False):
    os.makedirs(outdir, exist_ok=True)

    if use_3dpw:
        ckpt_file = 'data/vibe_data/vibe_model_w_3dpw.pth.tar'
        url = 'https://www.dropbox.com/s/41ozgqorcp095ja/vibe_model_w_3dpw.pth.tar'
        if not os.path.isfile(ckpt_file):
            download_url(url=url, outdir=outdir)
    else:
        ckpt_file = 'data/vibe_data/vibe_model_wo_3dpw.pth.tar'
        url = 'https://www.dropbox.com/s/amj2p8bmf6g56k6/vibe_model_wo_3dpw.pth.tar'
        if not os.path.isfile(ckpt_file):
            download_url(url=url, outdir=outdir)

    return ckpt_file


def images_to_video(img_folder, output_vid_file, img_suffix='%06d.png', img_list=None):
    # os.makedirs(img_folder, exist_ok=True)

    if img_list is not None:
        command = [
            'ffmpeg', '-y', '-threads', '16', '-f', 'concat', '-i', f'{img_list}', '-profile:v', 'baseline',
            '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
        ]
    else:
        command = [
            'ffmpeg', '-y', '-threads', '16', '-i', f'{img_folder}/{img_suffix}', '-profile:v', 'baseline',
            '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
        ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam


def convert_crop_coords_to_orig_img(bbox, keypoints, crop_size):
    # import IPython; IPython.embed(); exit()
    cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]

    # unnormalize to crop coords
    keypoints = 0.5 * crop_size * (keypoints + 1.0)

    # rescale to orig img crop
    keypoints *= h[..., None, None] / crop_size

    # transform into original image coords
    keypoints[:,:,0] = (cx - h/2)[..., None] + keypoints[:,:,0]
    keypoints[:,:,1] = (cy - h/2)[..., None] + keypoints[:,:,1]
    return keypoints


def prepare_rendering_results(vibe_results, nframes):
    frame_results = [{} for _ in range(nframes)]
    for person_id, person_data in vibe_results.items():
        for idx, frame_id in enumerate(person_data['frame_ids']):
            frame_results[frame_id][person_id] = {
                'verts': person_data['verts'][idx],
                'cam': person_data['orig_cam'][idx],
                'joints2d': person_data['smpl_joints2d'][idx]
            }

    # naive depth ordering based on the scale of the weak perspective camera
    for frame_id, frame_data in enumerate(frame_results):
        # sort based on y-scale of the cam in original image coords
        sort_idx = np.argsort([v['cam'][1] for k,v in frame_data.items()])
        frame_results[frame_id] = OrderedDict(
            {list(frame_data.keys())[i]:frame_data[list(frame_data.keys())[i]] for i in sort_idx}
        )

    return frame_results


def concat_videos(video_list, output_f, horizontal=False):
    # ffmpeg -i 1.mp4 -i 2.mp4 -i 3.mp4 -filter_complex hstack=3 out.mp4

    type = f'hstack={len(video_list)}' if horizontal else f'vstack={len(video_list)}'

    command = [
        'ffmpeg', '-y', '-threads', '16', '-filter_complex', type, output_f, '-v', 'error'
    ]

    for x in video_list:
        command.append('-i')
        command.append(x)

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)