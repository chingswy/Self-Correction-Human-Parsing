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
# k_cls_idxs = [[5, 6, 7], [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
k_cls_idxs = [[12], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]]    # for case 567, 568
'''
0-Background, 1-Hat, 2-Hair, 3-Glove, 4-Sunglasses,
5-Upper Cloth, 6-Drees, 7-Coat, 8-Sock, 9-Pant,
10-Jumpsuit, 11-Scarf, 12-Skirt, 13-Face, 14-Left Leg
15-Right Leg, 16-Left Arm, 17-Right Arm, 18-Left Shoe, 19-Right Shoe
'''

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

    flow_video_imgs = torch.zeros(num_frames - 1, frames.shape[2], frames.shape[3], frames.shape[1])
    raw_flows = torch.zeros(2, frames.shape[2], frames.shape[3])
    for i in tqdm(range(num_frames - 1)):
        pre_imgs_batch = torch.stack([frames[i]]).to(device)
        post_imgs_batch = torch.stack([frames[i + 1]]).to(device)

        list_of_flows = model(pre_imgs_batch, post_imgs_batch)
        predicted_flows = list_of_flows[-1]
        raw_flows = predicted_flows.cpu().detach()[0]

        np.save(os.path.join(out_imgs_dir, str(i).zfill(6)), np.round(raw_flows.permute(1, 2, 0).numpy()).astype(np.int16))

        flow_imgs = flow_to_image(predicted_flows)
        flow_video_imgs[i] = flow_imgs.permute(0, 2, 3, 1).cpu()[0]

    write_video(video_path.replace('.mp4', '_of.mp4'), flow_video_imgs, fps_out)
    print('Optical flows video saving done.')

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

def generate_valid_masks(imgs_path, output_path, thres, kernel_size=3):
    os.makedirs(output_path, exist_ok=True)
    imgs_name_list = sorted(glob(os.path.join(imgs_path, '*.jpg')))
    for i in range(len(imgs_name_list)):
        img_name = imgs_name_list[i]
        img = cv2.imread(img_name)
        mask = (img[:, :, 0] >= thres).astype(np.uint8)
        mask *= (img[:, :, 0] >= thres).astype(np.uint8)
        mask *= (img[:, :, 0] >= thres).astype(np.uint8)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_dilate = cv2.dilate(mask, kernel)
        mask_erode = cv2.erode(mask_dilate, kernel)

        img = cv2.resize(img, (1024, 1024), cv2.INTER_MAX)
        cv2.imwrite(os.path.join(output_path, str(i).zfill(6) + '.png'), mask_erode)

def generate_valid_parsing(parsings_path, masks_path, num_cls):
    out_dir = parsings_path.replace('opted', 'opted-thres')
    os.makedirs(out_dir, exist_ok=True)
    parsings_name_list = sorted(glob(os.path.join(parsings_path, '*.png')))
    for parsing_name in parsings_name_list:
        img_name = parsing_name.split(os.sep)[-1]
        mask_name = os.path.join(masks_path, img_name)
        
        parsing = cv2.imread(parsing_name, 2)
        mask = cv2.imread(mask_name, 2)
        
        parsing[((parsing > 0) - ((parsing > 0) * mask > 0).astype(np.uint8) > 0).astype(np.bool8)] = num_cls + 1

        out_name = parsing_name.replace('opted', 'opted-thres')
        cv2.imwrite(out_name, parsing)

def generate_opted_parsing(data_path, output_path, num_parser_cls, cls_idxs, is_thres=False):
    palette = get_palette(num_parser_cls)
    imgs_path_list = sorted(glob(os.path.join(data_path, 'images', '*')))
    for imgs_path in imgs_path_list:
        tmp_output_path = os.path.abspath(os.path.join(output_path, 'tmp_' + '_'.join(imgs_path.split(os.sep)[-3:])))
        os.makedirs(tmp_output_path, exist_ok=True)

        seq = imgs_path.split(os.sep)[-3]
        sub = imgs_path.split(os.sep)[-1]

        parsings_path = os.path.join(output_path, seq, 'mask-schp-parsing', sub)
        generate_merged_parsing(parsings_path, cls_idxs, palette)
        parsings_path = parsings_path.replace('parsing', 'merged-parsing')

        if os.path.exists(tmp_output_path + '/seq.mp4'):
            print('Already exsit the video of captured images!')
        else:
            generate_videos(imgs_path)
            mv_cmd = 'mv ' + imgs_path + '.mp4 ' + tmp_output_path + '/seq.mp4'
            os.system(mv_cmd)

        if os.path.exists(tmp_output_path + '/seq_of.mp4'):
            print('Already exsit optical flows of the video!')
        else:
            generate_optical_flows(tmp_output_path + '/seq.mp4')

        ofs_path = (tmp_output_path + '/seq.mp4').replace('.mp4', '_of/')
        optim_parsing(parsings_path, ofs_path, len(cls_idxs))
        parsings_path  = parsings_path.replace('parsing', 'parsing-opted')

        if is_thres:
            valid_masks_path = os.path.join(output_path, seq, 'mask-color-thres-close', sub)
            generate_valid_masks(imgs_path, valid_masks_path, 28)

            generate_valid_parsing(parsings_path, valid_masks_path, len(cls_idxs))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='')
    parser.add_argument('--output', type=str, default='data')
    parser.add_argument('--thres', action='store_true')
    args = parser.parse_args()

    generate_opted_parsing(args.path, args.output, k_num_parser_cls, k_cls_idxs, args.thres)