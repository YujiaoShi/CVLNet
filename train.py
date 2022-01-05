#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os

import torchvision.utils

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataLoader.DataLoad import load_data, load_train_data, load_test_grd_data, load_test_sat_data1
from dataLoader.datasets import train_file, val_file, test_file
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # for downloading pretrained VGG weights

# from FuseNet import FuseNet  # for project grd
from FuseModel import FuseModel
from FuseNet import FuseModelImg

from partical_similarity_loss import similarity_uncertainty, loss_uncertainty
# from SiamFuseNet import CrossLocalizationNet

from losses_for_training import HER_TriLoss_OR_UnNorm

import numpy as np
import os
import argparse

from utils import gps2distance
from val import RankVal, RankTest1, parse_args, getSavePath


def RankTrain(lr, args, save_path, writer):
    # if args.FCANET:
    #     criterion = HER_TriLoss_OR_UnNorm()
    # else:
    criterion = loss_uncertainty(args.shift_range)
    criterion.cuda()

    get_similarity_fn = similarity_uncertainty(args.shift_range)  # for test , in cpu

    bestRankResult = 0.0  # current best, Siam-FCANET18
    # loop over the dataset multiple times
    for epoch in range(args.resume, args.epochs):
        net.train()

        # base_lr = 0
        base_lr = lr
        base_lr = base_lr * ((1.0 - float(epoch) / 100.0) ** (1.0))

        print(base_lr)

        ###
        # optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
        optimizer = optim.Adam(net.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00005)

        optimizer.zero_grad()

        ### feeding A and P into train loader
        trainloader = load_data(train_file, mini_batch, args.stereo, args.sequence,
                                      args.shift_range, args.polar_sat, use_project_grd=0,
                                use_semantic=args.use_semantic)

        loss_vec = []

        print('batch_size:', mini_batch, '\n num of batches:', len(trainloader))

        for Loop, TripletData in enumerate(trainloader, 0):

            sat_map, left_camera_k, right_camera_k, grd_left_imgs, grd_right_imgs, \
            loc_shift_left, loc_shift_right, heading = [item.cuda() for item in TripletData[:-3]]

            # For visualization
            # sat_map.requires_grad = True
            # grd_left_imgs.requires_grad = True
            # sat_map.retain_grad()
            # grd_left_imgs.retain_grad()

            # zero the parameter gradients
            optimizer.zero_grad()

            if args.debug:
                file_name = TripletData[-1]
                print(file_name)
                out_dir = './visualize/'
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                B, S, _, _, _ = grd_left_imgs.shape
                for idx in range(B):
                    sat_img = transforms.functional.to_pil_image(sat_map[idx], mode='RGB')
                    sat_img.save(os.path.join(out_dir, 'sat_B' + str(idx) + '.png'))

                    for seq_idx in range(S):
                        grd_img = transforms.functional.to_pil_image(grd_left_imgs[idx, seq_idx], mode='RGB')
                        grd_img.save(os.path.join(out_dir, 'grd_left_B' + str(idx) + '_S' + str(seq_idx) + '.png'))


            grd_global, sat_global, uncertainty = net.forward(sat_map, left_camera_k, right_camera_k, grd_left_imgs,
                                                          grd_right_imgs, loc_shift_left,
                                                          loc_shift_right, heading)
            marginCal = 0.15
            if args.uncertainty:
                loss, real_pos_dist, pred_pos_dist = criterion(sat_global, grd_global, uncertainty)
            else:
                loss, real_pos_dist, pred_pos_dist = criterion(sat_global, grd_global)

            # if 0:
            #     B, S, _, _, _ = grd_left_imgs.shape
            #     uncer = 1 / uncertainty
            #     for idx in range(B):
            #         uncer[idx] = (uncer[idx] - torch.min(uncer[idx])) / (torch.max(uncer[idx]) - torch.min(uncer[idx]))
            #
            #     pad_uncertainty = F.pad(uncer, (13, 13, 13, 13))
            #     upsample = F.upsample(pad_uncertainty, sat_map.shape[-2:], mode='nearest')
            #     out_dir = './visualize/'
            #     if not os.path.exists(out_dir):
            #         os.makedirs(out_dir)
            #
            #
            #     from pytorch_grad_cam.grad_cam_utils.image import show_cam_on_image
            #     from PIL import Image
            #
            #     for idx in range(B):
            #         sat_img = transforms.functional.to_pil_image(sat_map[idx], mode='RGB')
            #         sat_img.save(os.path.join(out_dir, 'sat_B' + str(idx) + '.png'))
            #
            #         uncer_img = show_cam_on_image(sat_map[idx].data.cpu().numpy().transpose(1, 2, 0),
            #                                       upsample[idx, 0].data.cpu().numpy(), use_rgb=True)
            #         uncer_img = Image.fromarray(uncer_img)
            #         uncer_img.save(os.path.join(out_dir, 'uncer' + str(idx) + '.png'))
            #
            #         for seq_idx in range(S):
            #             grd_img = transforms.functional.to_pil_image(grd_left_imgs[idx, seq_idx], mode='RGB')
            #             grd_img.save(os.path.join(out_dir, 'grd_left_B' + str(idx) + '_S' + str(seq_idx) + '.png'))

            loss.backward()

            # !!! .step() weight = weight - learning_rate * gradient
            optimizer.step()  # This step is responsible for updating weights

            # print statistics
            running_loss = float(loss.data)
            p_dist = real_pos_dist.detach().cpu().numpy()
            n_dist = pred_pos_dist.detach().cpu().numpy()

            ### record the loss
            loss_vec.append(loss)

            if Loop % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch, Loop, running_loss))
                print('real positive distance 01: ', p_dist)
                print('pred positive distance 01: ', n_dist)
                writer.add_scalar('training loss', torch.mean(torch.stack(loss_vec, dim=-1)),
                                  epoch * len(trainloader) + Loop)
                # for b_idx in range(1):
                #     writer.add_image(str(b_idx) + 'grd_left:' + TripletData[-1][b_idx], grd_left_imgs[b_idx, 0, :, :, :])
                #     writer.add_image(str(b_idx) + 'sat_img:' + TripletData[-1][b_idx], sat_map[b_idx, :, :, :])
                #     writer.add_image(str(b_idx) + 'grd_feat:' + TripletData[-1][b_idx], F.upsample(grd_global[b_idx:b_idx+1], (512, 512), mode='nearest')[0])
                #     writer.add_image(str(b_idx) + 'sat_feat:' + TripletData[-1][b_idx],
                #                  F.upsample(sat_global[b_idx:b_idx + 1], (512, 512), mode='nearest')[0])
                #     writer.add_graph(net, inputs)
                #     for name, param in net.state_dict().items():
                #         writer.add_histogram(name, param)

                left_img_grid = torchvision.utils.make_grid(grd_left_imgs[:, 0, :, :, :], normalize=True)
                sat_img_grid = torchvision.utils.make_grid(sat_map, normalize=True)
                grd_feat_grid = torchvision.utils.make_grid(F.upsample(grd_global, (512, 512), mode='nearest'))
                sat_feat_grid = torchvision.utils.make_grid(F.upsample(sat_global, (512, 512), mode='nearest'))

                # ele_map_grid = torchvision.utils.make_grid(F.upsample(ele_map, (512, 512), mode='nearest'))

                writer.add_image('grd_left_imgs', left_img_grid)
                writer.add_image('sat_images', sat_img_grid)
                writer.add_image('grd_feature', grd_feat_grid)
                writer.add_image('sat_feature', sat_feat_grid)
                # writer.add_image('ele_map', ele_map_grid)
                # for b_idx in range(mini_batch):
                #     writer.add_text('file_name' + str(b_idx), TripletData[-1][b_idx])

        ### save modelget_similarity_fn
        compNum = epoch % 100
        print('taking snapshot ...')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if 0:
            torch.save(net.module.state_dict(), save_path + 'model_' + str(compNum) + '.pth')
        else:
            torch.save(net.state_dict(), os.path.join(save_path, 'model_' + str(compNum) + '.pth'))
        # np.save(save_path + 'loss_vec' + np.str(epoch) + 'epoch.npy', loss_vec)

        ### ranking test
        # current = RankVal(net, get_similarity_fn, args, save_path, bestRankResult)
        # if (current > bestRankResult):
        #     bestRankResult = current
        # np.save(save_path + 'model_' + str(epoch+1) + '.npy', current)
        #
        # print('')
        RankVal(epoch, net, get_similarity_fn, args, save_path, 0.)
        RankTest1(epoch, net, get_similarity_fn, args, save_path, 0.)

    print('Finished Training')



if __name__ == '__main__':
    # test to load 1 data
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    np.random.seed(2022)

    args = parse_args()

    mini_batch = args.batch_size

    restore_path, save_path = getSavePath(args)

    writer = SummaryWriter(save_path)

    if args.debug:
        net = FuseModel(debug_flag=args.debug, sequence=args.sequence, stereo=args.stereo, feature_win=512,
                       # height_planes=args.height_planes,
                       sim=args.sim, fuse_method=args.fuse_method,
                        # ele_order=args.ele_order,
                        seq_order=args.seq_order,
                       # height_sample=args.height_sample,
                        shift_range=args.shift_range,
                        proj=args.proj)
    else:

        if args.project_grd:
            net = FuseModelImg(debug_flag=args.debug, sequence=args.sequence, stereo=args.stereo, feature_win=32,
                            # height_planes=args.height_planes,
                            sim=args.sim, fuse_method=args.fuse_method,
                            seq_order=args.seq_order,
                            # height_sample=args.height_sample,
                            shift_range=args.shift_range,
                            proj=args.proj)
        else:

            net = FuseModel(debug_flag=args.debug, sequence=args.sequence, stereo=args.stereo, feature_win=32,
                           # height_planes=args.height_planes,
                           sim=args.sim, fuse_method=args.fuse_method,
                            seq_order=args.seq_order,
                           # height_sample=args.height_sample,
                            shift_range=args.shift_range,
                            proj=args.proj)

    ### cudaargs.epochs, args.debug)
    net.cuda()
    ###########################

    if args.test:
        net.load_state_dict(torch.load(os.path.join(save_path, 'Model_best.pth')))
        # net.load_state_dict(torch.load('./Models/sequence4_stereo0_fuse0_corrTrue_batch8_loss1_GRUdir2_GRUlayers1/stage_1/Model_best.pth'))
        get_similarity_fn = similarity_uncertainty(args.shift_range)  # for test , in cpu
        RankVal(net, get_similarity_fn, args, save_path, 0.)

    else:

        if args.resume:
            # net.load_state_dict(torch.load(os.path.join(save_path, 'model_0.pth')))

            net.load_state_dict(torch.load(os.path.join(save_path, 'model_' + str(args.resume - 1) + '.pth')))
            print("resume from " + 'model_' + str(args.resume - 1) + '.pth')
            lr = args.lr
            # start_epoch = args.resume

        else:

            if restore_path:
                save_dict = torch.load(os.path.join(restore_path, 'Model_best.pth'))
                net_dict = net.state_dict()
                state_dict = {k: v for k, v in save_dict.items() if
                              k in net_dict.keys() and net_dict[k].size() == save_dict[k].size()}
                net.load_state_dict(state_dict, strict=False)

                # net.load_state_dict(torch.load(os.path.join(restore_path, 'Model_best.pth')), strict=False)
                print('load model from ', os.path.join(restore_path, 'Model_best.pth'))

            if args.stage == 1:
                for param in net.SatFeatureNet.parameters():
                    param.requires_grad = False
                for param in net.SatDownch.parameters():
                    param.requires_grad = False
                for param in net.GrdFeatureNet.parameters():
                    param.requires_grad = False
                for param in net.GrdDownch.parameters():
                    param.requires_grad = False

                for param in net.UncertaintyNet.parameters():
                    param.requires_grad = False

                lr = 1e-4

            if args.stage == 2:
                for param in net.SatFeatureNet.parameters():
                    param.requires_grad = False
                for param in net.SatDownch.parameters():
                    param.requires_grad = False
                for param in net.UncertaintyNet.parameters():
                    param.requires_grad = False

                lr = 1e-4

            else:
                lr = args.lr

        RankTrain(lr, args, save_path, writer)
        writer.flush()
        writer.close()

