"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os
from PIL import Image

from .models import compile_model
from .data import compile_data
from .tools import SimpleLoss, get_batch_iou, get_val_info


def train(version,
            dataroot='/data/nuscenes',
            nepochs=10000,
            gpuid=1,

            H=900, W=1600,
            resize_lim=(0.193, 0.225),
            #这里将resize_lim改为max(128/480, 352/640)，
            #resize_lim=(0.52, 0.58),
            final_dim=(128, 352),
            bot_pct_lim=(0.0, 0.22),
            #小样本拟合时关闭随机旋转
            rot_lim=(-5.4, 5.4),
            #rot_lim=(0, 0),
            
            #小样本拟合时关闭随机翻转
            rand_flip=True,

            ncams=4,
            max_grad_norm=5.0,
            pos_weight=4,
            logdir='./runs',

            xbound=[-3, 3, 0.05],
            ybound=[-3, 3, 0.05],
            zbound=[-1.0, 1.0, 2.0],
            dbound=[0.5, 2.5, 0.05],

            bsz=2,
            nworkers=10,
            lr=1e-3,
            weight_decay=1e-7,
            ):
    #定义3d网格配置
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    #定义数据集增强配置（旋转，裁剪）
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_LEFT', 'CAM_FRONT', 'CAM_RIGHT',
                              'CAM_BACK'],
                    'Ncams': ncams,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)
    val_step = 1000 if version == 'mini' else 10000

    model.train()
    counter = 0
    for epoch in range(nepochs):
        np.random.seed()
        #bsz = 2,每次调用两次get_item,imgs的形状为(2, C, H, W……)，所以一共20个sample，会循环10次
        #nepochs=10000,所以会循环10万次，couter一直到100000
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs,ego_pose) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()
            preds = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
            binimgs = binimgs.to(device)
            #打印binimgs并查看
            # img0 = (binimgs[0, 0].cpu().numpy() * 255).astype(np.uint8)
            # Image.fromarray(img0).save(f"binimgs{batchi}_0.png")

            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print(counter, loss.item())
                _, _, iou = get_batch_iou(preds, binimgs)
                print('iou', iou)
                writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                
                val_info = get_val_info(model, valloader, loss_fn, device)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()
