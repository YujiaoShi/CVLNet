#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch

visualize_debug = 0 # 0: not debug, 1: get right position files, 2: visualize hitmap 3: visualize visibility 4: visualize vis grad

from FuseNet import FuseModelImg
from FuseModel import FuseModel
    
from dataLoader.DataLoad import load_test_grd_data, load_test_sat_data1, load_test_sat_data2

from partical_similarity_loss import similarity_uncertainty

import numpy as np

import scipy.io as scio
import time



########################### ranking test ############################
def RankTest(net_test, args, sat_batch, sequence, test_wo_destractors=True):

        
    get_similarity_fn = similarity_uncertainty(args.shift_range)#HER_TriLoss_OR_UnNorm() partical_similarity_loss()
    get_similarity_fn.cuda()

    grdloader = load_test_grd_data(mini_batch, args.stereo, sequence,
                              use_project_grd=0)

    ### init for restoring the features
    query_vec = torch.tensor([]) #np.zeros([N_data,vec_len], dtype=np.float32)
    grd_location_vec = torch.tensor([])  # np.zeros([N_data,vec_len], dtype=np.float32)

    if test_wo_destractors:
        satloader1 = load_test_sat_data1(sat_batch, True, args.polar_sat)
        sat_vec1 = torch.tensor([]) #np.zeros([N_data,vec_len], dtype=np.float32)
        sat_location_vec1 = torch.tensor([]) #np.zeros([N_data,vec_len], dtype=np.float32)
        uncertainty_vec1 = torch.tensor([])

    satloader2 = load_test_sat_data2(sat_batch, args.polar_sat)
    sat_vec2 = torch.tensor([])  # np.zeros([N_data,vec_len], dtype=np.float32)
    sat_location_vec2 = torch.tensor([])  # np.zeros([N_data,vec_len], dtype=np.float32)
    uncertainty_vec2 = torch.tensor([])


    ### grd feature extract
    if os.path.exists(os.path.join(save_path, str(sequence) + 'query.mat')):
        data = scio.loadmat(os.path.join(save_path, str(sequence) + 'query.mat'))
        query_vec = data['query_vec']
        query_vec = torch.from_numpy(query_vec)

        grd_location_vec = data['grd_location_vec']
        grd_location_vec = torch.from_numpy(grd_location_vec)

        print('load query vector from mat done: ', os.path.join(save_path, str(sequence) + 'query.mat') )

    else:

        grd_start_time = time.time()

        for i, data in enumerate(grdloader, 0):

            left_camera_k, right_camera_k, grd_left_imgs, grd_right_imgs, \
            loc_shift_left, loc_shift_right, heading, loc_left = [item.cuda() for item in data[:-1]]

            outputs_query, _, _ = net_test.forward(None, left_camera_k, right_camera_k,
                                                                   grd_left_imgs, grd_right_imgs, loc_shift_left,
                                                                   loc_shift_right, heading)

            ###### feature vector feeding
            query_vec = torch.cat([query_vec,outputs_query.data.cpu()],dim=0) # [count,1024]
            grd_location_vec = torch.cat([grd_location_vec, loc_left[:, 0, :2].cpu()], dim=0)  # [count,2]

            del outputs_query, left_camera_k, right_camera_k, grd_left_imgs, grd_right_imgs, \
            loc_shift_left, loc_shift_right, heading, loc_left, data

        duration = time.time() - grd_start_time

        print('feature extraction time for grd images, sequence ' + str(sequence) + ' second per image: ', duration/len(grdloader))
        scio.savemat(os.path.join(save_path, str(sequence) +'query.mat'),
                     {'query_vec': query_vec.data.cpu().numpy(),
                      'grd_location_vec': grd_location_vec.data.cpu().numpy()})
        print('save query vector done')

    ### sat feature extract
    if test_wo_destractors:
        if os.path.exists(os.path.join(save_path, 'sat1.mat')):
            data = scio.loadmat(os.path.join(save_path, 'sat1.mat'))
            sat_vec1 = data['sat_vec1']
            sat_vec1 = torch.from_numpy(sat_vec1)

            sat_location_vec1 = data['sat_location_vec1']
            sat_location_vec1 = torch.from_numpy(sat_location_vec1)

            uncertainty_vec1 = data['uncertainty_vec1']
            uncertainty_vec1 = torch.from_numpy(uncertainty_vec1)
            print('load sat vector from mat done: ', os.path.join(save_path, 'sat1.mat'))

        else:
            for i, data in enumerate(satloader1, 0):
                sat_map, loc_sat = data

                sat_location_vec1 = torch.cat([sat_location_vec1, loc_sat],dim=0) # [count,2]

                sat_map = sat_map.cuda()

                _, outputs_sat_vec1, uncertainty = net_test.forward(sat_map, None, None, None, None, None, None, None)

                ###### feature vector feeding
                sat_vec1 = torch.cat([sat_vec1,outputs_sat_vec1.data.cpu()], dim=0)
                uncertainty_vec1 = torch.cat([uncertainty_vec1, uncertainty.data.cpu()], dim=0)

                del outputs_sat_vec1, uncertainty, loc_sat, sat_map, data


            print(sat_vec1.data.cpu().numpy().shape)
            scio.savemat(os.path.join(save_path, 'sat1.mat'),
                         {'sat_vec1': sat_vec1.data.cpu().numpy(),
                          'sat_location_vec1': sat_location_vec1.cpu().numpy(),
                          'uncertainty_vec1': uncertainty_vec1.cpu().numpy()})
            print('save sat1 vector done')

    if os.path.exists(os.path.join(save_path, 'sat2.mat')):
        data = scio.loadmat(os.path.join(save_path, 'sat2.mat'))
        sat_vec2 = data['sat_vec2']
        sat_vec2 = torch.from_numpy(sat_vec2)

        sat_location_vec2 = data['sat_location_vec2']
        sat_location_vec2 = torch.from_numpy(sat_location_vec2)

        uncertainty_vec2 = data['uncertainty_vec2']
        uncertainty_vec2 = torch.from_numpy(uncertainty_vec2)
        print('load sat vector from mat done: ', os.path.join(save_path, 'sat2.mat'))

    else:

        sat2_start_time = time.time()

        for i, data in enumerate(satloader2, 0):
            sat_map, loc_sat = data

            sat_location_vec2 = torch.cat([sat_location_vec2, loc_sat], dim=0)  # [count,2]

            sat_map = sat_map.cuda()

            _, outputs_sat_vec2, uncertainty = net_test.forward(sat_map, None, None, None, None, None, None, None)

            ###### feature vector feeding
            sat_vec2 = torch.cat([sat_vec2, outputs_sat_vec2.data.cpu()], dim=0)
            uncertainty_vec2 = torch.cat([uncertainty_vec2, uncertainty.data.cpu()], dim=0)

            del outputs_sat_vec2, uncertainty, loc_sat, sat_map, data

        duration = time.time() - sat2_start_time
        print('feature extraction time for sat2 images, second per image: ',
              duration / len(satloader2))

        scio.savemat(os.path.join(save_path, 'sat2.mat'),
                     {'sat_vec2': sat_vec2.data.cpu().numpy(),
                      'sat_location_vec2': sat_location_vec2.cpu().numpy(),
                      'uncertainty_vec2': uncertainty_vec2.cpu().numpy()})
        print('save sat2 vector done')
    print('vec produce done')
        
    ### load vectors    
    N_data = query_vec.shape[0]

    f = open(os.path.join(save_path, str(sequence) + 'test_results.txt'), 'a')

    if test_wo_destractors:
        print('without distraction')
        f.write('------------- without distraction -----------------\n')
        M_data = sat_vec1.shape[0]

        if os.path.exists(os.path.join(save_path, str(sequence) + 'dist1.mat')):
            data = scio.loadmat(os.path.join(save_path, str(sequence) + 'dist1.mat'))
            dist_array = data['dist_array']
            dist_array = torch.from_numpy(dist_array)

            shift_meters_matrix = data['shift_meters_matrix']
            shift_meters_matrix = torch.from_numpy(shift_meters_matrix)

            print('load dist.mat from ', os.path.join(save_path, str(sequence) + 'dist1.mat'))

        else:
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
                    similarity, shift_meters = get_similarity_fn(query_vec[start_i:end_i].cuda(),
                                                                 sat_vec1[start_j:end_j].cuda(),
                                                                 uncertainty_vec1[start_j: end_j].cuda())  # ,test_method)
                    similarity_sat_matrix = torch.cat([similarity_sat_matrix, similarity.cpu()], dim=0)
                    shift_meters_sat_matrix = torch.cat([shift_meters_sat_matrix, shift_meters.cpu()], dim=0)
                similarity_matrix = torch.cat([similarity_matrix, similarity_sat_matrix], dim=1)
                shift_meters_matrix = torch.cat([shift_meters_matrix, shift_meters_sat_matrix], dim=1)  # [M, N, 2]
                del similarity, shift_meters

            dist_array = 2 - 2 * similarity_matrix

            print(dist_array.data.cpu().numpy().shape)
            scio.savemat(os.path.join(save_path, str(sequence) + 'dist1.mat'),
                         {'dist_array': dist_array.data.cpu().numpy(),
                          'shift_meters_matrix': shift_meters_matrix.data.cpu().numpy()})
            print('save dist array done')

        prediction_id = torch.topk(dist_array, 100, dim=0, largest=False, sorted=True)[1]  # [100, N]

        results = []

        for topk in (1, 5, 10, 100):
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
                print('median: ' + str(median_dis) + ' mean: ' + str(mean_dis) + ' std: ' + str(std_dis))
                print('------------------------------------------------------------')
                f.write('============================================================')
                f.write('median: ' + str(median_dis) + ' mean: ' + str(mean_dis) + ' std: ' + str(std_dis))
                f.write('------------------------------------------------------------')

            line = ''
            meter = 10
            correct_num = torch.sum(torch.le(min_dis, meter))
            result = float(correct_num) / float(N_data)
            results.append(result)

            line += str(result * 100) + ' '

            print('top-' + str(topk) + ' within 10 meters: ' + line)
            f.write('top-' + str(topk) + ' within 10 meters: ' + line + '\n')

        results = np.array(results).reshape(2, 4)
        out_path = os.path.join(save_path, str(sequence) + 'test1_results')
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        scio.savemat(os.path.join(out_path, str(sequence) + 'results1.mat'), {'results': results,
                                                                                  'prediction_id': prediction_id,
                                                                                  'dist_array': dist_array,
                                                                                  })

    f.write('------------- with distraction -----------------\n')
    M_data = sat_vec2.shape[0]
    
    retrieval2_start_time = time.time()

    if os.path.exists(os.path.join(save_path, str(sequence) + 'dist2.mat')):
        data = scio.loadmat(os.path.join(save_path, str(sequence) + 'dist2.mat'))
        dist_array = data['dist_array']
        dist_array = torch.from_numpy(dist_array)

        shift_meters_matrix = data['shift_meters_matrix']
        shift_meters_matrix = torch.from_numpy(shift_meters_matrix)

        print('load dist.mat from ', os.path.join(save_path, str(sequence) + 'dist2.mat'))

    else:
        similarity_matrix = torch.tensor([])
        shift_meters_matrix = torch.tensor([])
        batch = 50
        for i in range(int(np.ceil(N_data/batch))):
            start_i = i*batch
            end_i = start_i + min(batch,N_data+1-start_i)
            similarity_sat_matrix = torch.tensor([])
            shift_meters_sat_matrix = torch.tensor([])
            for j in range(int(np.ceil(M_data/batch))):
                start_j = j*batch
                end_j = start_j + min(batch,M_data+1-start_j)
                similarity, shift_meters = get_similarity_fn(query_vec[start_i:end_i].cuda(),
                                                             sat_vec2[start_j:end_j].cuda(),
                                               uncertainty_vec2[start_j: end_j].cuda())# ,test_method)
                similarity_sat_matrix = torch.cat([similarity_sat_matrix, similarity.cpu()], dim=0)
                shift_meters_sat_matrix = torch.cat([shift_meters_sat_matrix, shift_meters.cpu()], dim=0)
            similarity_matrix = torch.cat([similarity_matrix, similarity_sat_matrix], dim=1)
            shift_meters_matrix = torch.cat([shift_meters_matrix, shift_meters_sat_matrix], dim=1)  # [M, N, 2]
            del similarity, shift_meters

        dist_array = 2-2*similarity_matrix

        print(dist_array.data.cpu().numpy().shape)
        scio.savemat(os.path.join(save_path, str(sequence) + 'dist2.mat'),
                     {'dist_array': dist_array.data.cpu().numpy(),
                      'shift_meters_matrix': shift_meters_matrix.data.cpu().numpy()})
        print('save dist array done')

    prediction_id = torch.topk(dist_array, 100, dim=0, largest=False, sorted=True)[1] #[100, N]

    duration = time.time() - retrieval2_start_time
    print('retrieval time: ', duration / dist_array.shape[1])

    results = []


    for topk in (1,5,10,100):
        min_dis = None

        for j in range(topk):
            # grd_x, grd_y = gps2utm_torch(grd_location_vec[:, 0], grd_location_vec[:, 1]) # [N]
            # sat_x, sat_y = gps2utm_torch(sat_location_vec[prediction_id[j, :], 0],
            #                                 sat_location_vec[prediction_id[j, :], 1])
            grd_x, grd_y = grd_location_vec[:, 0], grd_location_vec[:, 1]
            sat_x, sat_y = sat_location_vec2[prediction_id[j, :], 0], sat_location_vec2[prediction_id[j, :], 1]
            sat_x = sat_x + shift_meters_matrix[prediction_id[j, :], np.arange(0, N_data), 1]
            sat_y = sat_y + shift_meters_matrix[prediction_id[j, :], np.arange(0, N_data), 0]
            dis = torch.sqrt((sat_x - grd_x)**2 + (sat_y - grd_y)**2)
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
            print('median: ' + str(median_dis) + ' mean: ' + str(mean_dis) + ' std: ' + str(std_dis))
            print('------------------------------------------------------------')
            f.write('============================================================')
            f.write('median: ' + str(median_dis) + ' mean: ' + str(mean_dis) + ' std: ' + str(std_dis))
            f.write('------------------------------------------------------------')

        line = ''
        meter = 10
        correct_num = torch.sum(torch.le(min_dis, meter))
        result = float(correct_num) / float(N_data)
        results.append(result)

        line += str(result*100) + ' '

        print('top-' + str(topk) + ' within 10 meters: ' + line)
        f.write('top-' + str(topk) + ' within 10 meters: ' + line + '\n')

    results = np.array(results).reshape(4, 2)
    out_path = os.path.join(save_path, str(sequence) + 'test2_results')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    scio.savemat(os.path.join(out_path, str(sequence) + 'results2.mat'), {'results': results,
                                                         'prediction_id': prediction_id,
                                                         'dist_array': dist_array,
                                                         })

    return
##########################

from train import parse_args, getSavePath


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # test

    batch_count = 32

    args = parse_args()
    _, save_path = getSavePath(args)
    mini_batch = args.batch_size

    print('same_test_sequence:', args.same_test_sequence)


    if args.project_grd:
        net = FuseModelImg(debug_flag=args.debug, sequence=args.sequence, stereo=args.stereo, feature_win=32,
                           # height_planes=args.height_planes,
                           sim=args.sim, fuse_method=args.fuse_method,
                           seq_order=args.seq_order,
                           # height_sample=args.height_sample,
                           shift_range=args.shift_range,
                           proj=args.proj)

        net.cuda()

        net.load_state_dict(torch.load(os.path.join(save_path, 'Model_best.pth')))
        print("restore finished")

        RankTest(net, args, batch_count, args.sequence)

    else:


        if args.same_test_sequence:
            sequence = args.sequence
            print(sequence)
            net = FuseModel(debug_flag=args.debug, sequence=args.sequence, stereo=args.stereo, feature_win=32,
                            # height_planes=args.height_planes,
                            sim=args.sim, fuse_method=args.fuse_method,
                            seq_order=args.seq_order,
                            # height_sample=args.height_sample,
                            shift_range=args.shift_range)

            net.cuda()

            net.load_state_dict(torch.load(os.path.join(save_path, 'Model_best.pth')), strict=False)
            print("restore finished")

            RankTest(net, args, batch_count, sequence, test_wo_destractors=True)

        else:
            # for sequence in reversed(range(1, 17)):
            for sequence in [1, 4]:
                print(sequence, '/16')
                net = FuseModel(debug_flag=args.debug, sequence=sequence, stereo=args.stereo, feature_win=32,
                                # height_planes=args.height_planes,
                                sim=args.sim, fuse_method=args.fuse_method,
                                seq_order=args.seq_order,
                                # height_sample=args.height_sample,
                                shift_range=args.shift_range)
                net.cuda()

                net.load_state_dict(torch.load(os.path.join(save_path, 'Model_best.pth')), strict=False)
                print("restore finished")

                RankTest(net, args, batch_count, sequence, False)


