import os
import cv2
import torch

import numpy as np
import torchvision.transforms.functional as F

from tqdm import tqdm
from glob import glob

from PIL.ImagePalette import ImagePalette
from torchvision.io import read_video
from torchvision.io import write_video
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image

from mhp_extension.logits_fusion import get_palette
from easymocap.visualize.ffmpeg_wrapper import VideoMaker

k_num_parser_cls = 20
k_cls_idxs = [[5, 6, 7], [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]

def generate_videos(imgs_path, restart=True, fps_in=50, fps_out=50, remove_imgs=False, reorder=False,
                    ext='.jpg', debug=False):
    if os.path.exists(imgs_path + '.mp4'):
        print('Already exsit video of captured images!')
        return

    video_maker = VideoMaker(restart=restart, fps_in=fps_in, fps_out=fps_out, remove_images=remove_imgs,
                             reorder=reorder, ext=ext, debug=debug)
    video_maker.make_video(imgs_path)

def generate_optical_flows(video_path, device='cuda', fps_out=50):
    out_imgs_dir = video_path.replace('.mp4', '_of/')
    os.makedirs(out_imgs_dir, exist_ok=True)

    frames, _, _ = read_video(video_path)
    frames = frames.permute(0, 3, 1, 2).float() / 255.
    num_frames = frames.shape[0]

    model = raft_large(pretrained=True, progress=True).to(device)
    model = model.eval()

    # batch resize
    # frames = F.resize(frames, size=[512, 512], antialias=False)

    # flow_video_imgs = torch.zeros(num_frames - 1, frames.shape[2], frames.shape[3], frames.shape[1])
    raw_flows = torch.zeros(2, frames.shape[2], frames.shape[3])
    for i in tqdm(range(num_frames - 1)):
        pre_imgs_batch = torch.stack([frames[i]]).to(device)
        post_imgs_batch = torch.stack([frames[i + 1]]).to(device)

        list_of_flows = model(pre_imgs_batch, post_imgs_batch)
        predicted_flows = list_of_flows[-1]
        raw_flows = predicted_flows.cpu().detach()[0]

        np.save(os.path.join(out_imgs_dir, str(i).zfill(6)), np.round(raw_flows.permute(1, 2, 0).numpy()).astype(np.int16))

        # flow_imgs = flow_to_image(predicted_flows)
        # flow_video_imgs[i] = flow_imgs.permute(0, 2, 3, 1).cpu()[0]

    # write_video(video_path.replace('.mp4', '_of.mp4'), flow_video_imgs, fps_out)
    print('Optical flow video saving done.')

def inverse_palette(palette, color):
    img_palette = ImagePalette(mode='RGB', palette=palette)
    idx = img_palette.getcolor(color)
    return idx

def merge_parsing(img, cls_idxs, palette):
    merged_parsing = np.zeros_like(img[:, :, 0])
    for i_cls in range(len(cls_idxs)):
        i_cls_mask = np.zeros_like(img[:, :, 0])
        for j_cls in cls_idxs[i_cls]:
            j_cls_r = palette[j_cls * 3 + 0]
            j_cls_g = palette[j_cls * 3 + 1]
            j_cls_b = palette[j_cls * 3 + 2]

            mask = np.array(img[:, :, 0] == j_cls_r)
            mask *= np.array(img[:, :, 1] == j_cls_g)
            mask *= np.array(img[:, :, 2] == j_cls_b)
            i_cls_mask += np.array(mask, dtype=np.uint8)

        merged_parsing += i_cls_mask * (i_cls + 1)
    return merged_parsing

def generate_merged_parsing(imgs_path, cls_idxs, palette):
    imgs_name = sorted(glob(os.path.join(imgs_path, '*.png')))
    out_dir = imgs_path.replace('parsing', 'merged-parsing')
    os.makedirs(out_dir, exist_ok=True)
    for img_name in tqdm(imgs_name):
        img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        merged_parsing = merge_parsing(img, cls_idxs, palette)
        out_name = img_name.replace('parsing', 'merged-parsing')
        cv2.imwrite(out_name, merged_parsing)

def evaluate_optical_flow(pre_frame, optical_flow, mode='forward'):
    h, w = optical_flow.shape[:2]
    warp_grid_x, warp_grid_y = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))
    if mode == 'forward':
        flow_inv = np.stack((warp_grid_x, warp_grid_y), axis=-1) - optical_flow
    elif mode == 'backward':
        flow_inv = np.stack((warp_grid_x, warp_grid_y), axis=-1) + optical_flow
    flow_inv = flow_inv.astype(np.float32)
    post_frame = cv2.remap(pre_frame, flow_inv, None, cv2.INTER_LINEAR)
    return post_frame

def optim_parsing(parsings_path, optical_flows_path, num_cls):
    imgs_name = sorted(glob(os.path.join(parsings_path, '*.png')))
    ofs_name = sorted(glob(os.path.join(optical_flows_path, '*.npy')))
    out_dir = parsings_path.replace('parsing', 'parsing-opted')
    os.makedirs(out_dir, exist_ok=True)
    for i in tqdm(range(1, len(imgs_name) - 1)):
        pre_parsing = cv2.imread(imgs_name[i-1])
        parsing = cv2.imread(imgs_name[i])
        post_parsing = cv2.imread(imgs_name[i+1])

        pre_optical_flow = np.load(ofs_name[i-1])
        post_optical_flow = np.load(ofs_name[i])

        pre_parsing = evaluate_optical_flow(pre_parsing, pre_optical_flow, mode='forward')
        post_parsing = evaluate_optical_flow(post_parsing, post_optical_flow, mode='backward')

        opted_parsing = np.zeros_like(parsing)
        for j in range(num_cls):
            cls_mask = (parsing == (j + 1)).astype(np.uint8)
            pre_cls_mask = (pre_parsing == (j + 1)).astype(np.uint8)
            post_cls_mask = (post_parsing == (j + 1)).astype(np.uint8)
            cls_mask = cls_mask + pre_cls_mask + post_cls_mask
            cls_mask = (cls_mask >= 2).astype(np.uint8)
            opted_parsing += cls_mask * (j + 1)

        valid_mask = (parsing > 0)
        opted_parsing *= valid_mask

        out_name = imgs_name[i].replace('parsing', 'parsing-opted')
        cv2.imwrite(out_name, opted_parsing)

if __name__ == '__main__':
    # palette = get_palette(k_num_parser_cls)
    '''
    0-Background, 1-Hat, 2-Hair, 3-Glove, 4-Sunglasses,
    5-Upper Cloth, 6-Drees, 7-Coat, 8-Sock, 9-Pant,
    10-Jumpsuit, 11-Sarf, 12-Skirt, 13-Face, 14-Left Leg
    15-Right Leg, 16-Left Arm, 17-Right Arm, 18-Left Shoe, 19-Right Shoe
    '''
    # parsings_path = '/datasets/shen/ClothProj/results/schp/CoreView_550_01/mask-schp-parsing/01'
    # generate_merged_parsing(parsings_path, k_cls_idxs, palette)

    # imgs_path = '/datasets/shen/ClothProj/data/zju_mocap/CoreView_550_01/images/01'
    # generate_videos(imgs_path)
    # generate_optical_flows(imgs_path + '.mp4')

    # pre_path = '/datasets/shen/ClothProj/data/zju_mocap/CoreView_550_01/images/01/000128.jpg'
    # of_path = '/datasets/shen/ClothProj/data/zju_mocap/CoreView_550_01/images/01_of/000128.npy'
    # pre_frame = cv2.imread(pre_path)
    # optical_flow = np.load(of_path)
    # post_frame = evaluate_optical_flow(pre_frame, optical_flow)
    # cv2.imwrite('/datasets/shen/ClothProj/data/zju_mocap/CoreView_550_01/images/01_of/000128_post.jpg', post_frame)

    parsings_path = '/datasets/shen/ClothProj/results/schp/CoreView_550_01/mask-schp-merged-parsing/01'
    optical_flows_path = '/datasets/shen/ClothProj/data/zju_mocap/CoreView_550_01/images/01_of'
    optim_parsing(parsings_path, optical_flows_path, len(k_cls_idxs))