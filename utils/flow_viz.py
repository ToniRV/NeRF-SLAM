#!/usr/bin/env python3
# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np
from icecream import ic

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    # TODO: Don't compute colorwheel every time...
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False, flow_norm=None):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]

    if flow_norm is not None:
        assert flow_norm > 0
    else:
        rad = np.sqrt(np.square(u) + np.square(v))
        rad_max = np.max(rad)
        epsilon = 1e-5
        flow_norm = rad_max + epsilon

    u = u / flow_norm
    v = v / flow_norm

    return flow_uv_to_colors(u, v, convert_to_bgr)


import cv2
import torch
import torch.nn.functional as F

def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def upsample_flow(flow, mask):
    """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
    N, _, H, W = flow.shape
    mask = mask.view(N, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(8 * flow, [3,3], padding=1)
    up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, 2, 8*H, 8*W)

def cvx_upsample(data, mask, pow=1.0):
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd) # for each pixel (ht,wd) we have 9 pixels (3x3) surrounding it and we upsample the 8x8 sub-pixels.
    # Set all borders to 0, so we do cvx combination of valid pixels only, otw you'll do cvx combinations of 0-valued pixels, which neither for depth nor covariance is valid...
    mask[:, :, [[0,1,2], [6,7,8]], :, :, [[0],[-1]], :] = -torch.inf # for the top of img (first row, all cols)
    mask[:, :, [[0,3,6], [2,5,8]], :, :, :, [[0],[-1]]] = -torch.inf # for the top of img (first row, all cols)
    mask = torch.softmax(mask, dim=2).pow(pow)

    up_data = F.unfold(data, [3,3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)

    return up_data

def upsample_depth(depth, mask):
    batch, num, ht, wd = depth.shape
    depth = depth.view(batch*num, ht, wd, 1)
    mask = mask.view(batch*num, -1, ht, wd)
    return cvx_upsample(depth, mask).view(batch, num, 8*ht, 8*wd)


def viz_flow(name, flow, mask=None, flow_norm=None):
    if mask is None:
        # Uncomment to visualize delta flow (in motion estimation)
        # up_flow = upflow8(flow[:,0].permute(0,3,1,2))
        up_flow = upflow8(flow[None], mode="bicubic")
    else:
        # Won't really work since the mask is for 1D depths not for 2D flows... although...
        #up_flow = upsample_flow(flow[:,0].permute(0,3,1,2), mask)
        up_flow = cvx_upsample(flow.permute(0,3,1,2), mask)
    up_flow = up_flow.squeeze().permute(1,2,0).cpu().numpy()

    # Use Hue, Saturation, Value colour model 
    mag, ang = cv2.cartToPolar(up_flow[..., 0], up_flow[..., 1])
    mask = mag > 700
    mag[mask] = 700 # clip the flow that is too large

    hsv = np.zeros((up_flow.shape[0], up_flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255 # Saturation
    hsv[mask,1] = 0 # color white the flow that is too large
    hsv[..., 0] = ang * 180 / np.pi / 2 # Hue
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) # Value

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    cv2.imshow(name, rgb)
    return rgb


def viz_flow_sigma(flow_cov, img_bg=None):
    valid = (flow_cov > 0.0).all(dim=-1).all(dim=-1)

    H, W = flow_cov.shape[0], flow_cov.shape[1]
    # Only visualize eigen vectors
    S = torch.linalg.svdvals(flow_cov.view(H*W, 2, 2)).view(H, W, 2)
    # Calculate the max norm of the sigma eigen values, and normalize them...
    # not the nicest way to visualize the ellipsoids for the optical flow... but hey it is something
    # This viz is only useful to compare eigen values in the image, since by normalizing among
    # the max eigen value in the image, we can only see how strong is a pixel's flow eigen value wrt to other pixels' flows.
    # Well in principle S_min, gives you a sense of what is a good conditioned covariance...
    S_max = (S[...,0] / S[...,0].max().item()) # to make it from 0 to 1
    S_min = (S[...,1] / S[...,1].max().item())

    S_max[~valid] = 1.0 # max covariance for invalid flows
    S_min[~valid] = 1.0

    S_max = S_max[None,None]
    S_min = S_min[None,None]

    new_size = (8 * S_max.shape[-2], 8 * S_max.shape[-1])
    flow_sigma_x = F.interpolate(S_max, size=new_size, mode="bilinear", align_corners=True)
    flow_sigma_y = F.interpolate(S_min, size=new_size, mode="bilinear", align_corners=True)
    flow_sigma_x = flow_sigma_x.squeeze().cpu()
    flow_sigma_y = flow_sigma_y.squeeze().cpu()

    flow_sigma_x = (flow_sigma_x.cpu().numpy()*255).astype(np.uint8) # scaling from 16bit to 8bit
    flow_sigma_y = (flow_sigma_y.cpu().numpy()*255).astype(np.uint8) # scaling from 16bit to 8bit

    flow_sigma_x = cv2.applyColorMap(flow_sigma_x, cv2.COLORMAP_JET)
    flow_sigma_y = cv2.applyColorMap(flow_sigma_y, cv2.COLORMAP_JET)

    if img_bg is not None:
        img_bg = img_bg.permute(1,2,0).cpu().numpy()
        flow_sigma_x = cv2.addWeighted(img_bg, 0.5, flow_sigma_x, 0.5, 0)
        flow_sigma_y = cv2.addWeighted(img_bg, 0.5, flow_sigma_y, 0.5, 0)

    cv2.imshow("Up flow Sigma eigen_max", flow_sigma_x)
    cv2.imshow("Up flow Sigma eigen_min", flow_sigma_y)



def viz_idepth(disps, upmask, fix_range=False):
    disps_up = cvx_upsample(disps.unsqueeze(-1), upmask).squeeze().unsqueeze(-1)
    depth = 1.0 / disps_up[-1].squeeze().to(torch.float) # Visualize only the last depth...
    viz_depth_map(depth, fix_range)

def viz_depth_map(depth, fix_range=False, name='Depth up', invert=True, colormap=cv2.COLORMAP_PLASMA, write=False):
    valid = (depth > 0) & (~torch.isinf(depth)) & (~torch.isnan(depth)) # one shouldn't use exact equality on floats but for synthetic values it's ok
    if fix_range:
        dmin, dmax = 0.0, 3.0
    else:
        if valid.any():
            dmin = depth[valid].min().item()
            dmax = depth[valid].max().item()
        else:
            dmin = depth.min().item()
            dmax = depth.max().item()
    output = (depth - dmin) / (dmax - dmin) # dmin -> 0.0, dmax -> 1.0
    output[output < 0] = 0 # saturate
    output[output > 1] = 1 # saturate
    #output = output / (output + 0.1)# 0 -> white, 1 -> black
    if invert:
        output = 1.0 - output # 0 -> white, 1 -> black
    output[~valid] = 1.0 if invert else 0.0 # black out invalid
    #output = output.pow(1/2) # most picture data is gamma-compressed
    output = (output.cpu().numpy()*255).astype(np.uint8) # scaling from 16bit to 8bit
    output = cv2.applyColorMap(output, colormap)
    cv2.imshow(name, output)
    if write:
        cv2.imwrite(name+".png", output)

def viz_idepth_sigma(idepth_cov, upmask, fix_range=False, bg_img=None, write=False):
    img_id = -1

    idepth_sigma = idepth_cov.sqrt()
    idepth_sigma_up = cvx_upsample(idepth_sigma.unsqueeze(-1), upmask).squeeze().unsqueeze(-1)
    idepth_sigma_up = idepth_sigma_up[img_id].squeeze().to(torch.float) # Visualize only the last depth...

    valid = (idepth_sigma_up > 0) # one shouldn't use exact equality on floats but for synthetic values it's ok
    dmin = idepth_sigma_up.min().item()
    if fix_range:
        dmax = 10.0
    else:
        dmax = idepth_sigma_up.max().item()
    output = (idepth_sigma_up - dmin) / (dmax - dmin) # dmin -> 0.0, dmax -> 1.0
    output[output < 0] = 0 # saturate
    output[output > 1] = 1 # saturate
    #output = output / (output + 0.1)# 0 -> white, 1 -> black
    output[~valid] = 1 # black out invalid
    #output = output.pow(1/2) # most picture data is gamma-compressed
    output = (output.cpu().numpy()*255).astype(np.uint8) # scaling from 16bit to 8bit
    output = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    if bg_img is not None:
        bg_img = bg_img[img_id].permute(1,2,0).cpu().numpy()
        output = cv2.addWeighted(bg_img, 0.5, output, 0.5, 0)
    name = 'Inverse Depth Sigma up'
    cv2.imshow(name, output)
    if write:
        cv2.imwrite(name+".png", output)

# Depth: [9, 480, 640, 1]
# Img:   [9, 3, 480, 640])
def viz_depth_sigma(depth_sigma_up, fix_range=False, bg_img=None, name='Depth Sigma up', sigma_thresh=10.0, write=True):
    img_id = -1

    #ic(depth_sigma_up.shape)
    #ic(bg_img.shape)

    depth_sigma_up_viz = depth_sigma_up[img_id].squeeze().to(torch.float) # Visualize only the last depth...

    valid = (depth_sigma_up_viz > 0) # one shouldn't use exact equality on floats but for synthetic values it's ok
    dmin = depth_sigma_up_viz.min().item()
    if fix_range:
        dmax = sigma_thresh
    else:
        dmax = depth_sigma_up_viz.max().item()
    output = (depth_sigma_up_viz - dmin) / (dmax - dmin) # dmin -> 0.0, dmax -> 1.0
    output[output < 0] = 0 # saturate
    output[output > 1] = 1 # saturate
    #output = output / (output + 0.1)# 0 -> white, 1 -> black
    output[~valid] = 1 # black out invalid
    #output = output.pow(1/2) # most picture data is gamma-compressed
    output = (output.cpu().numpy()*255).astype(np.uint8) # scaling from 16bit to 8bit
    output = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    if bg_img is not None:
        bg_img = bg_img[img_id].permute(1,2,0).cpu().numpy()
        output = cv2.addWeighted(bg_img, 0.5, output, 0.5, 0)
    cv2.imshow(name, output)
    if write:
        cv2.imwrite(name+".png", output)


def viz_img_and_flow(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    # map flow to rgb image
    flo = flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
