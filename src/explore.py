"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""
import os
import numpy as np
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches

from .data import compile_data
from .tools import (ego_to_cam, get_only_in_img_mask, denormalize_img,
                    SimpleLoss, get_val_info, add_ego, gen_dx_bx,
                    get_nusc_maps, plot_nusc_map)
from .models import compile_model




def lidar_check(version,
                dataroot='/data/nuscenes',
                show_lidar=True,
                viz_train=False,
                nepochs=1,

                H=900, W=1600,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,

                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[4.0, 45.0, 1.0],

                bsz=1,
                nworkers=10,
                ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': cams,
                    'Ncams': 5,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='vizdata')

    loader = trainloader if viz_train else valloader

    model = compile_model(grid_conf, data_aug_conf, outC=1)

    rat = H / W
    val = 10.1
    fig = plt.figure(figsize=(val + val/3*2*rat*3, val/3*2*rat))
    gs = mpl.gridspec.GridSpec(2, 6, width_ratios=(1, 1, 1, 2*rat, 2*rat, 2*rat))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    for epoch in range(nepochs):
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, pts, binimgs) in enumerate(loader):

            img_pts = model.get_geometry(rots, trans, intrins, post_rots, post_trans)

            for si in range(imgs.shape[0]):
                plt.clf()
                final_ax = plt.subplot(gs[:, 5:6])
                for imgi, img in enumerate(imgs[si]):
                    ego_pts = ego_to_cam(pts[si], rots[si, imgi], trans[si, imgi], intrins[si, imgi])
                    mask = get_only_in_img_mask(ego_pts, H, W)
                    plot_pts = post_rots[si, imgi].matmul(ego_pts) + post_trans[si, imgi].unsqueeze(1)

                    ax = plt.subplot(gs[imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    plt.imshow(showimg)
                    if show_lidar:
                        plt.scatter(plot_pts[0, mask], plot_pts[1, mask], c=ego_pts[2, mask],
                                s=5, alpha=0.1, cmap='jet')
                    # plot_pts = post_rots[si, imgi].matmul(img_pts[si, imgi].view(-1, 3).t()) + post_trans[si, imgi].unsqueeze(1)
                    # plt.scatter(img_pts[:, :, :, 0].view(-1), img_pts[:, :, :, 1].view(-1), s=1)
                    plt.axis('off')

                    plt.sca(final_ax)
                    plt.plot(img_pts[si, imgi, :, :, :, 0].view(-1), img_pts[si, imgi, :, :, :, 1].view(-1), '.', label=cams[imgi].replace('_', ' '))
                
                plt.legend(loc='upper right')
                final_ax.set_aspect('equal')
                plt.xlim((-50, 50))
                plt.ylim((-50, 50))

                ax = plt.subplot(gs[:, 3:4])
                plt.scatter(pts[si, 0], pts[si, 1], c=pts[si, 2], vmin=-5, vmax=5, s=5)
                plt.xlim((-50, 50))
                plt.ylim((-50, 50))
                ax.set_aspect('equal')

                ax = plt.subplot(gs[:, 4:5])
                plt.imshow(binimgs[si].squeeze(0).T, origin='lower', cmap='Greys', vmin=0, vmax=1)

                imname = f'lcheck{epoch:03}_{batchi:05}_{si:02}.jpg'
                print('saving', imname)
                plt.savefig(imname)


def cumsum_check(version,
                dataroot='/data/nuscenes',
                gpuid=1,

                H=900, W=1600,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,

                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[4.0, 45.0, 1.0],

                bsz=4,
                nworkers=10,
                ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': 6,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
    loader = trainloader

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    model.eval()
    for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):

        model.use_quickcumsum = False
        model.zero_grad()
        out = model(imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
                )
        out.mean().backward()
        print('autograd:    ', out.mean().detach().item(), model.camencode.depthnet.weight.grad.mean().item())

        model.use_quickcumsum = True
        model.zero_grad()
        out = model(imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
                )
        out.mean().backward()
        print('quick cumsum:', out.mean().detach().item(), model.camencode.depthnet.weight.grad.mean().item())
        print()


def eval_model_iou(version,
                modelf,
                dataroot='/data/nuscenes',
                gpuid=0,

                H=900, W=1600,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,

                xbound=[-3, 3, 0.05],
                ybound=[-3, 3, 0.05],
                zbound=[-1.0, 1.0, 2.0],
                dbound=[0.5, 2.5, 0.05],

                bsz=2,
                nworkers=10,
                ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_LEFT', 'CAM_FRONT', 'CAM_RIGHT',
                              'CAM_BACK'],
                    'Ncams': 4,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    print('loading', modelf)
    model.load_state_dict(torch.load(modelf))
    model.to(device)

    loss_fn = SimpleLoss(1.0).cuda(gpuid)

    model.eval()
    val_info = get_val_info(model, valloader, loss_fn, device)
    print(val_info)


def viz_model_preds(version,
                    modelf,
                    dataroot='/data/nuscenes',
                    map_folder='/data/nuscenes/mini',
                    gpuid=0,
                    viz_train=False,

                    H=900, W=1600,
                    resize_lim=(0.193, 0.225),
                    final_dim=(128, 352),
                    bot_pct_lim=(0.0, 0.22),
                    rot_lim=(-5.4, 5.4),
                    rand_flip=True,

                    xbound=[-3, 3, 0.05],
                    ybound=[-3, 3, 0.05],
                    zbound=[-1.0, 1.0, 2.0],
                    dbound=[0.5, 2.5, 0.05],

                    bsz=2,
                    nworkers=10,
                    ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_LEFT', 'CAM_FRONT', 'CAM_RIGHT',
            'CAM_BACK']
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': cams,
                    'Ncams': 4,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')
    loader = trainloader if viz_train else valloader
    #nusc_maps = get_nusc_maps(map_folder)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    print('loading', modelf)
    model.load_state_dict(torch.load(modelf))
    model.to(device)

    dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    # scene2map = {}
    # for rec in loader.dataset.nusc.scene:
    #     log = loader.dataset.nusc.get('log', rec['log_token'])
    #     scene2map[rec['name']] = log['location']


    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    model.eval()
    counter = 0
    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs,ego_poses) in enumerate(loader):
            out = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
            out = out.sigmoid().cpu()
            # img0 = (out[0, 0].numpy() * 255).astype(np.uint8)
            # Image.fromarray(img0).save("out_000.png")

            #imgs对应一个批次，bsz = 2,一个批次有两个sample,所以imgs.shape[0]是2
            #这里遍历一个批次的两个sample,si = 0,1
            for si in range(imgs.shape[0]):
                plt.clf()
                #遍历每个批次的前后左右四张图片，并显示
                for imgi, img in enumerate(imgs[si]):
                    #划分网格3*3，前三排放分割图，先不放，这里在后两排放拍摄的图，
                    #中左-左，中中-前，中右-右，后中-后
                    if imgi == 0:
                        ax = plt.subplot(gs[1, 0])  # 第二行 左
                    elif imgi == 1:
                        ax = plt.subplot(gs[1, 1])  # 第二行 中
                    elif imgi == 2:
                        ax = plt.subplot(gs[1, 2])  # 第二行 右
                    elif imgi == 3:
                        ax = plt.subplot(gs[2, 1])  # 第三行 中 ← 你想放的位置
                    #反归一化
                    showimg = denormalize_img(img)
                    #把朝后拍摄的图片反转，好看
                    if imgi > 2:
                        showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                    #显示图片，关闭坐标轴，添加图例
                    plt.imshow(showimg)
                    plt.axis('off')
                    plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction')
                #清空第一排，用以放地图和分割结果
                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                plt.setp(ax.spines.values(), color='b', linewidth=2)
                #绘制图例
                plt.legend(handles=[
                    mpatches.Patch(color=(0.0, 0.0, 1.0, 1.0), label='Output Vehicle Segmentation'),
                    mpatches.Patch(color='#76b900', label='Ego Vehicle'),
                    mpatches.Patch(color=(1.00, 0.50, 0.31, 0.8), label='Map (for visualization purposes only)')
                ], loc=(0.01, 0.86))
                
                
                ##########################原始加载背景图#########################
                # # plot static map (improves visualization)
                # rec = loader.dataset.ixes[counter]
                # plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                # plt.xlim((out.shape[3], 0))
                # plt.ylim((0, out.shape[3]))
                #add_ego(bx, dx)
                ########################以下为新的加载背景图#######################
                # 加载全局背景图


                this_dir = os.path.dirname(os.path.abspath(__file__))
                map_path = os.path.join(this_dir, "..", "mydataset", "mini", "maps", "global_map.png")
                global_map = Image.open(map_path).convert("RGB")

                # 每米多少像素
                scale = 25  # 1000px / 40m
                map_size_px = 1000
                map_size_m = 40

                # 获取无人机在全局地图中的位置（单位：米）
                ux, uy = ego_poses[si][0], ego_poses[si][1]
                print("ux:", ux, "uy:", uy)
                # 将 (x, y) 世界坐标转换为像素坐标
                # 原点 (0,0) 在图像中心，所以要加偏移
                center = map_size_px // 2
                px = int(center + uy * scale)  # 这是因为opencv的x坐标方向向右，在airsim里这是y的方向
                py = int(center - ux * scale)  # 注意不仅x,y轴交换，y轴的图像方向向下，所以要减

                # 截取范围（半径为6m/2=3m → 3*25=75px）
                half_crop = int(3 * scale)
             
                left = px - half_crop
                top = py - half_crop
                right = px + half_crop
                bottom = py + half_crop

                # 防止越界
                left = max(0, left)
                top = max(0, top)
                right = min(map_size_px, right)
                bottom = min(map_size_px, bottom)
                
                if right <= left or bottom <= top:
                    print(f"[WARNING] Invalid crop area: left={left}, right={right}, top={top}, bottom={bottom}")
                    break  # 或 return 或 break，根据你的逻辑决定
                # 裁剪图像
                # 裁剪出背景图
                cropped_map = global_map.crop((left, top, right, bottom))

                # 调整为和 BEV 输出一样的分辨率（比如 200x200）
                bev_H, bev_W = out.shape[2], out.shape[3]
                background_img = cropped_map.resize((bev_W, bev_H))
                # cropped_map.save(f'cropped_map_{batchi:06}_{si:03}.png')         # 保存裁剪前的地图局部
                # background_img.save(f'background_img_{batchi:06}_{si:03}.png')   # 保存缩放后的地图局部（BEV尺寸）


                # 显示在背景图上
                plt.imshow(background_img, extent=[0, bev_W, bev_H, 0])

                #将自身显示在背景图上
                add_ego(bx, dx)

                #一个批次的一个sample对应一个out,这里把每个out(语义分割图)，值限幅在0-1并用蓝色输出,这里设置一下透明度，保证背景和预测标准同时能够显示
                plt.imshow(out[si].squeeze(0), vmin=0, vmax=1, cmap='Blues',alpha=0.6)

                ################################################################
                imname = f'eval{batchi:06}_{si:03}.jpg'
                print('saving', imname)
                plt.savefig(imname)
                counter += 1
