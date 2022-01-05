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

from FuseNet import FuseModelImg  # for project grd
from FuseModel import FuseModel

from partical_similarity_loss import similarity_uncertainty, loss_uncertainty
# from SiamFuseNet import CrossLocalizationNet

from losses_for_training import HER_TriLoss_OR_UnNorm

import numpy as np
import os
import argparse

from utils import gps2distance
# from train import

########################### ranking test ############################
def RankTest1(epoch, net_test, get_similarity_fn, args, save_path, best_rank_result):
    ### net evaluation state
    net_test.eval()
    mini_batch = args.batch_size

    grdloader = load_test_grd_data(mini_batch, args.stereo, args.sequence,
                                   use_project_grd=0,
                                use_semantic=args.use_semantic)

    ### init for restoring the features
    query_vec = torch.tensor([])  # np.zeros([N_data,vec_len], dtype=np.float32)
    grd_location_vec = torch.tensor([])  # np.zeros([N_data,vec_len], dtype=np.float32)


    satloader1 = load_test_sat_data1(mini_batch, True, args.polar_sat)
    sat_vec1 = torch.tensor([])  # np.zeros([N_data,vec_len], dtype=np.float32)
    sat_location_vec1 = torch.tensor([])  # np.zeros([N_data,vec_len], dtype=np.float32)
    uncertainty_vec1 = torch.tensor([])

    for i, data in enumerate(grdloader, 0):
        left_camera_k, right_camera_k, grd_left_imgs, grd_right_imgs, \
        loc_shift_left, loc_shift_right, heading, loc_left = [item.cuda() for item in data[:-1]]

        outputs_query, _, _ = net_test.forward(None, left_camera_k, right_camera_k,
                                               grd_left_imgs, grd_right_imgs, loc_shift_left,
                                               loc_shift_right, heading,
                                               attn_pdrop=0, resid_pdrop=0, pe_pdrop=0)

        ###### feature vector feeding
        query_vec = torch.cat([query_vec, outputs_query.data.cpu()], dim=0)  # [count,1024]
        grd_location_vec = torch.cat([grd_location_vec, loc_left[:, 0, :2].cpu()], dim=0)  # [count,2]

        del outputs_query, left_camera_k, right_camera_k, grd_left_imgs, grd_right_imgs, \
            loc_shift_left, loc_shift_right, heading, loc_left, data

    for i, data in enumerate(satloader1, 0):
        sat_map, loc_sat = data

        sat_location_vec1 = torch.cat([sat_location_vec1, loc_sat], dim=0)  # [count,2]

        sat_map = sat_map.cuda()

        _, outputs_sat_vec1, uncertainty = net_test.forward(sat_map, None, None, None, None, None, None, None,
                                               attn_pdrop=0, resid_pdrop=0, pe_pdrop=0)

        ###### feature vector feeding
        sat_vec1 = torch.cat([sat_vec1, outputs_sat_vec1.data.cpu()], dim=0)
        uncertainty_vec1 = torch.cat([uncertainty_vec1, uncertainty.data.cpu()], dim=0)

        del outputs_sat_vec1, uncertainty, loc_sat, sat_map, data

    ### load vectors
    N_data = query_vec.shape[0]
    M_data = sat_vec1.shape[0]

    similarity_matrix = torch.tensor([])
    shift_meters_matrix = torch.tensor([])
    batch = 50
    for i in range(int(np.ceil(N_data / batch))):
        start_i = i * batch
        end_i = start_i + min(batch, N_data + 1 - start_i)
        similarity_sat_matrix = torch.tensor([])
        shift_meters_sat_matrix = torch.tensor([])
        for j in range(int(np.ceil(M_data / batch))):
            start_j = j * batch
            end_j = start_j + min(batch, M_data + 1 - start_j)
            if args.uncertainty:
                similarity, shift_meters = get_similarity_fn(query_vec[start_i:end_i].cuda(),
                                                             sat_vec1[start_j:end_j].cuda(),
                                                             uncertainty_vec1[start_j: end_j].cuda())
            else:
                similarity, shift_meters = get_similarity_fn(query_vec[start_i:end_i].cuda(),
                                                             sat_vec1[start_j:end_j].cuda())
            similarity_sat_matrix = torch.cat([similarity_sat_matrix, similarity.cpu()], dim=0)
            shift_meters_sat_matrix = torch.cat([shift_meters_sat_matrix, shift_meters.cpu()], dim=0)
        similarity_matrix = torch.cat([similarity_matrix, similarity_sat_matrix], dim=1)
        shift_meters_matrix = torch.cat([shift_meters_matrix, shift_meters_sat_matrix], dim=1)  # [M, N, 2]
        del similarity, shift_meters

        if i % 10 == 0:
            print(i)

    dist_array = 2 - 2 * similarity_matrix

    prediction_id = torch.topk(dist_array, 100, dim=0, largest=False, sorted=True)[1]  # [top_k, N]
    print(prediction_id.shape)

    f = open(os.path.join(save_path, 'test1_results'), 'a')
    results = []
    for topk in (1,5,10,100):
        min_dis = None

        for j in range(topk):
            # grd_x, grd_y = gps2utm_torch(grd_location_vec[:, 0], grd_location_vec[:, 1]) # [N]
            # sat_x, sat_y = gps2utm_torch(sat_location_vec[prediction_id[j, :], 0],
            #                                 sat_location_vec[prediction_id[j, :], 1])
            grd_x, grd_y = grd_location_vec[:, 0], grd_location_vec[:, 1]
            sat_x, sat_y = sat_location_vec1[prediction_id[j, :], 0], sat_location_vec1[prediction_id[j, :], 1]
            sat_x = sat_x + shift_meters_matrix[prediction_id[j, :], np.arange(0, N_data), 1]
            sat_y = sat_y + shift_meters_matrix[prediction_id[j, :], np.arange(0, N_data), 0]
            dis = torch.sqrt((sat_x - grd_x) ** 2 + (sat_y - grd_y) ** 2)
            # dis = gps2distance(grd_location_vec[:, 0], grd_location_vec[:, 1],
            #                    sat_location_vec[prediction_id[j, :], 0],
            #                    sat_location_vec[prediction_id[j, :], 1])
            # print('dis-----',j,dis)
            if min_dis != None:
                min_dis = torch.minimum(min_dis, dis)
            else:
                min_dis = dis

        if topk == 1:
            median_dis = torch.median(dis).numpy()
            std_dis, mean_dis = torch.std_mean(dis)
            std_dis, mean_dis = std_dis.numpy(), mean_dis.numpy()
            print('============================================================')
            print('Epoch: ' + str(epoch) + ' median: ' + str(median_dis) + ' mean: ' + str(mean_dis) + ' std: ' + str(std_dis))
            print('------------------------------------------------------------')
            f.write('============================================================' + '\n')
            f.write('Epoch: ' + str(epoch) + ' median: ' + str(median_dis) + ' mean: ' + str(mean_dis) + ' std: ' + str(std_dis) + '\n')
            f.write('------------------------------------------------------------' + '\n')

        line = ''
        # for meter in (10, 25):
        for meter in (10, 15, 20, 25):
            correct_num = torch.sum(torch.le(min_dis, meter))
            result = float(correct_num) / float(N_data)
            results.append(result)

            line += str(result * 100) + ' '

        print('top-' + str(topk) + ' within 10, 15, 20 and 25 meters: ' + line)
        f.write('top-' + str(topk) + ' within 10, 15, 20 and 25 meters: ' + line + '\n')

    return



def RankVal(epoch, net_test, get_similarity_fn, args, save_path, best_rank_result):
    ### net evaluation state
    net_test.eval()

    mini_batch = args.batch_size

    valloader = load_data(val_file, mini_batch, args.stereo, args.sequence,
                            args.shift_range, args.polar_sat, use_project_grd=0,
                                use_semantic=args.use_semantic)

    ### init for restoring the features
    query_vec = torch.tensor([])  # np.zeros([N_data,vec_len], dtype=np.float32)
    grd_location_vec = torch.tensor([])  # np.zeros([N_data,vec_len], dtype=np.float32)

    # satloader1 = load_test_sat_data1(mini_batch, True, args.polar_sat)
    sat_vec1 = torch.tensor([])  # np.zeros([N_data,vec_len], dtype=np.float32)
    sat_location_vec1 = torch.tensor([])  # np.zeros([N_data,vec_len], dtype=np.float32)
    uncertainty_vec1 = torch.tensor([])

    for i, data in enumerate(valloader, 0):
        sat_map, left_camera_k, right_camera_k, grd_left_imgs, grd_right_imgs, \
        loc_shift_left, loc_shift_right, heading, loc_left, loc_sat = [item.cuda() for item in data[:-1]]

        # left_camera_k, right_camera_k, grd_left_imgs, grd_right_imgs, \
        # loc_shift_left, loc_shift_right, heading, loc_left = [item.cuda() for item in data[:-1]]

        # outputs_query, _, _ = net_test.forward(None, left_camera_k, right_camera_k,
        #                                        grd_left_imgs, grd_right_imgs, loc_shift_left,
        #                                        loc_shift_right, heading)

        outputs_query, outputs_sat_vec1, uncertainty = net_test.forward(sat_map, left_camera_k, right_camera_k, grd_left_imgs,
                                                          grd_right_imgs, loc_shift_left,
                                                          loc_shift_right, heading,
                                                          attn_pdrop=0, resid_pdrop=0, pe_pdrop=0)

        ###### feature vector feeding
        query_vec = torch.cat([query_vec, outputs_query.data.cpu()], dim=0)  # [count,1024]
        grd_location_vec = torch.cat([grd_location_vec, loc_left[:, 0, :2].cpu()], dim=0)  # [count,2]

        sat_location_vec1 = torch.cat([sat_location_vec1, loc_sat.cpu()], dim=0)  # [count,2]
        sat_vec1 = torch.cat([sat_vec1, outputs_sat_vec1.data.cpu()], dim=0)
        uncertainty_vec1 = torch.cat([uncertainty_vec1, uncertainty.data.cpu()], dim=0)

        del outputs_query, left_camera_k, right_camera_k, grd_left_imgs, grd_right_imgs, \
            loc_shift_left, loc_shift_right, heading, loc_left, data
        del outputs_sat_vec1, uncertainty, loc_sat, sat_map

        if i % 10 == 0:
            print(i)

    ### load vectors
    N_data = query_vec.shape[0]
    M_data = sat_vec1.shape[0]

    similarity_matrix = torch.tensor([])
    shift_meters_matrix = torch.tensor([])
    batch = 50
    for i in range(int(np.ceil(N_data / batch))):
        start_i = i * batch
        end_i = start_i + min(batch, N_data + 1 - start_i)
        similarity_sat_matrix = torch.tensor([])
        shift_meters_sat_matrix = torch.tensor([])
        for j in range(int(np.ceil(M_data / batch))):
            start_j = j * batch
            end_j = start_j + min(batch, M_data + 1 - start_j)
            if args.uncertainty:
                similarity, shift_meters = get_similarity_fn(query_vec[start_i:end_i].cuda(),
                                                             sat_vec1[start_j:end_j].cuda(),
                                                             uncertainty_vec1[start_j: end_j].cuda())
            else:
                similarity, shift_meters = get_similarity_fn(query_vec[start_i:end_i].cuda(),
                                                             sat_vec1[start_j:end_j].cuda())
            similarity_sat_matrix = torch.cat([similarity_sat_matrix, similarity.cpu()], dim=0)
            shift_meters_sat_matrix = torch.cat([shift_meters_sat_matrix, shift_meters.cpu()], dim=0)
        similarity_matrix = torch.cat([similarity_matrix, similarity_sat_matrix], dim=1)
        shift_meters_matrix = torch.cat([shift_meters_matrix, shift_meters_sat_matrix], dim=1)  # [M, N, 2]
        del similarity, shift_meters

    dist_array = 2 - 2 * similarity_matrix

    prediction_id = torch.topk(dist_array, 100, dim=0, largest=False, sorted=True)[1]  # [top_k, N]
    print(prediction_id.shape)
    min_dis = None
    f = open(os.path.join(save_path, 'val_results'), 'a')
    topk = 1
    for topk in (1,5,10,100):
        min_dis = None
        for j in range(topk):
            # grd_x, grd_y = gps2utm_torch(grd_location_vec[:, 0], grd_location_vec[:, 1]) # [N]
            # sat_x, sat_y = gps2utm_torch(sat_location_vec[prediction_id[j, :], 0],
            #                                 sat_location_vec[prediction_id[j, :], 1])
            grd_x, grd_y = grd_location_vec[:, 0], grd_location_vec[:, 1]
            sat_x, sat_y = sat_location_vec1[prediction_id[j, :], 0], sat_location_vec1[prediction_id[j, :], 1]
            sat_x = sat_x + shift_meters_matrix[prediction_id[j, :], np.arange(0, N_data), 1]
            sat_y = sat_y + shift_meters_matrix[prediction_id[j, :], np.arange(0, N_data), 0]
            dis = torch.sqrt((sat_x - grd_x) ** 2 + (sat_y - grd_y) ** 2)
            # dis = gps2distance(grd_location_vec[:, 0], grd_location_vec[:, 1],
            #                    sat_location_vec[prediction_id[j, :], 0],
            #                    sat_location_vec[prediction_id[j, :], 1])
            # print('dis-----',j,dis)
            if min_dis != None:
                min_dis = torch.minimum(min_dis, dis)
            else:
                min_dis = dis

        if topk == 1:
            median_dis = torch.median(dis).numpy()
            std_dis, mean_dis = torch.std_mean(dis)
            std_dis, mean_dis = std_dis.numpy(), mean_dis.numpy()
            print('============================================================')
            print('Epoch: ' + str(epoch) + ' median: ' + str(median_dis) + ' mean: ' + str(mean_dis) + ' std: ' + str(std_dis))
            print('------------------------------------------------------------')

            f.write('============================================================'+'\n')
            f.write('Epoch: ' + str(epoch) + ' median: ' + str(median_dis) + ' mean: ' + str(mean_dis) + ' std: ' + str(std_dis) + '\n')
            f.write('------------------------------------------------------------'+'\n')

        line = ''
        # for meter in (10, 25):
        for meter in (10, 15, 20, 25):
            correct_num = torch.sum(torch.le(min_dis, meter))
            result = float(correct_num) / float(N_data)
            # results.append(result)

            line += str(result * 100) + ' '

        print('top-' + str(topk) + ' within 10, 15, 20, and 25 meters: ' + line)
        f.write('top-' + str(topk) + ' within 10, 15, 20, and 25 meters: ' + line + '\n')

    # meter = 10
    #
    # correct_num = torch.sum(torch.le(min_dis, meter))
    # result = float(correct_num) / float(N_data)
    # print('top-' + str(topk) + ' within ' + str(meter) + ' meters: ' + str(result * 100))
    # f.write('top-' + str(topk) + ' within ' + str(meter) + ' meters: ' + str(result * 100) + '\n')
        ### save the best params
    # if (result > best_rank_result):
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     torch.save(net_test.state_dict(), os.path.join(save_path, 'Model_best.pth'))
    #
    # f.write('top-' + str(args.top_k) + ' within ' + str(meter) + ' meters: ' + str(result * 100) + '\n')
    # f.close()

    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')
    parser.add_argument('--debug', type=int, default=0, help='debug to dump middle processing images')

    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')  # 1e-2

    parser.add_argument('--stereo', type=int, default=0, help='use left and right ground image')
    parser.add_argument('--sequence', type=int, default=4, help='use n images merge to 1 ground image')
    # parser.add_argument('--seq_turb', type=float, default=0., help='n% turblent on ortation and shift when loader sequence data')
    # parser.add_argument('--roUnknow', type=int, default=0, help='rotation of the compare image unknow')
    parser.add_argument('--shift_range', type=int, default=3, help='shift_pixel in correlation, '
                                                                   'if 0, does not apply shift')
    parser.add_argument('--height_planes', type=int, default=1, help='height layer, 1m * n')
    parser.add_argument('--height_sample', type=str, default='inverse', help='inverse or uniform')

    parser.add_argument('--gap_map', type=int, default=1, help='0:pair, 1:10m gap')

    parser.add_argument('--test_all', type=int, default=1, help='0:pair, 1: sparse, 2:all')
    parser.add_argument('--top_k', type=int, default=5, help='top k in result check')
    parser.add_argument('--right_range', type=int, default=25, help='right range in result check')

    parser.add_argument('--fuse_method', type=str, default='fuse_Transformer', help='vis_Conv2D, '
                                                                               'vis_LSTM, '
                                                                               'vis_Conv3D, '
                                                                               'fuse_LSTM, '
                                                                               'fuse_Conv3D'
                                                                               'fuse_Transformer')
    # parser.add_argument('--FCANET', type=int, default=0, help='FCANET')

    parser.add_argument('--seq_order', type=int, default=2, help='0:single direction starts from query location, '
                                                                 '1:single direction ends at query location,'
                                                                 '2:bidirectional')
    # parser.add_argument('--ele_order', type=int, default=2, help='0:single forward direction, '
    #                                                              '1:single inverse direction,'
    #                                                              '2:bidirectional')

    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    # parser.add_argument('--loss_method', type=int, default=2, help='0, 1, 2')

    # parser.add_argument('--use_corr', type=bool, default=True, help='use corr')
    parser.add_argument('--stage', type=int, default=0, help='0, 1, 2')
    # if stage is 0, model (for anyone) is trained from scratch, end-to-end training
    # if stage is 1, load pretrained model from single image trained model.  All pretrained weights are fixed.
    # This only applies for when sequence>1 or stereo=1
    # if stage is 2, fixed the satellite branch and finetune the ground branch

    parser.add_argument('--polar_sat', type=int, default=0, help='use polar changed satillite map  ')
    parser.add_argument('--project_grd', type=int, default=0, help='use projected ground image ')
    parser.add_argument('--sim', type=int, default=0, help='whether or not to use'
                                                           ' handcrafted similarity for visibility estimation ')

    parser.add_argument('--train_ignore', type=int, default=1, help='during training, ignore hard examples')
    parser.add_argument('--test_ignore', type=int, default=1, help='during testing, ignore hard examples')

    parser.add_argument('--uncertainty', type=int, default=1, help='with or without uncertainty')
    parser.add_argument('--proj', type=str, default='Geometry', help='Geometry, Unet, Reshape')

    parser.add_argument('--same_test_sequence', type=int, default=1, help='')

    parser.add_argument('--use_semantic', type=int, default=0, help='use semantic or not')

    args = parser.parse_args()

    return args


def getSavePath(args):

    if (args.sequence == 1 and args.stereo == 0):
        initial_path = './Models/FuseModel/sequence' + str(args.sequence) + '_stereo' + str(args.stereo) + \
                       '_corr' + str(args.shift_range) + \
                       '_batch' + str(args.batch_size) + '_uncertainty' + str(args.uncertainty)
        save_path = os.path.join(initial_path, 'stage_' + str(args.stage))
        if args.stage == 0:
            restore_path = None
        else:
            restore_path = os.path.join(initial_path, 'stage_0')

    else:

        initial_path = './Models/FuseModel/sequence' + str(args.sequence) + '_stereo' + str(args.stereo) + \
                       '_' + str(args.fuse_method) + '_corr' + str(args.shift_range) + \
                       '_batch' + str(args.batch_size) + '_uncertainty' + str(args.uncertainty) + \
                       '_seqOrder' + str(args.seq_order)
        save_path = os.path.join(initial_path, 'stage_' + str(args.stage))

        if args.stage==0:
            restore_path = None
        elif args.stage==1:
            path2 = os.path.join('./Models/FuseModel/sequence1_stereo0_corr' + str(args.shift_range) + \
                       '_batch' + str(args.batch_size), 'stage_2')

            path0 = os.path.join('./Models/FuseModel/sequence1_stereo0_corr' + str(args.shift_range) + \
                                 '_batch' + str(args.batch_size), 'stage_0')
            if os.path.exists(path2):
                restore_path = path2
            else:
                restore_path = path0
        elif args.stage==2:
            restore_path = os.path.join(initial_path, 'stage_0')

    if args.proj!='Geometry':
        if restore_path:
            restore_path = restore_path + '_' + args.proj 
        save_path = save_path + '_' + args.proj

    if args.project_grd:
        save_path = save_path.replace('/FuseModel/', '/FuseModelImg/')

    if args.use_semantic:
        save_path = save_path.replace('/FuseModel/', '/FuseModelSemantic/')

    print('restore_path:', restore_path)
    print('save_path:', save_path)

    return restore_path, save_path


if __name__ == '__main__':
    # test to load 1 data
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    np.random.seed(2022)

    args = parse_args()

    mini_batch = args.batch_size

    restore_path, save_path = getSavePath(args)

    writer = SummaryWriter(save_path)

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

    # for epoch in range(0, 5):
    #     net.load_state_dict(torch.load(os.path.join(save_path, 'model_' + str(epoch) + '.pth')))
    #     get_similarity_fn = similarity_uncertainty(args.shift_range)  # for test , in cpu
    #     # it seems non-neccessary to define in cpu or gpu, as there is no torch parameter in the function
    #     RankVal(epoch, net, get_similarity_fn, args, save_path, 0.)
    #     RankTest1(epoch, net, get_similarity_fn, args, save_path, 0.)

    net.load_state_dict(torch.load(os.path.join(save_path, 'Model_best.pth')), strict=False)
    get_similarity_fn = similarity_uncertainty(args.shift_range)  # for test , in cpu
    # it seems non-neccessary to define in cpu or gpu, as there is no torch parameter in the function
    # RankVal(0, net, get_similarity_fn, args, save_path, 0.)
    RankTest1(0, net, get_similarity_fn, args, save_path, 0.)


