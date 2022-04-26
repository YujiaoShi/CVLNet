

import torch.nn as nn
import torch.nn.functional as F
import torch

import utils


class partical_similarity(nn.Module):
    def __init__(self, shift_range):
        super(partical_similarity, self).__init__()
        self.shift_range = shift_range

    def forward(self, grd_feature, sat_feature, test_method=1):
        # test_method: 0:sqrt, 1:**2, 2:crop

        # use grd as kernel, sat map as inputs, convolution to get correlation
        # M, C, H_s, W_s = sat_feature.size()

        # _, _, H, W = grd_feature.size()
        # kernel_H = min(H, H_s - 2 * self.shift_range)  # 24: 38 meters ahead and 38 meters behind
        # kernel_W = min(W, W_s - 2 * self.shift_range)
        #
        # # only use kernel_edge in center as kernel, for efficient process time
        # W_start = W // 2 - kernel_W // 2
        # W_end = W // 2 + kernel_W // 2
        # H_start = H // 2 - kernel_H // 2
        # H_end = H // 2 + kernel_H // 2
        # grd_feature = grd_feature[:, :, H_start:H_end, W_start:W_end]
        N, C, H_k, W_k = grd_feature.size()
        # if test_method != 2: #not crop_method: # normalize later
        grd_feature = F.normalize(grd_feature.reshape(N, -1))
        grd_feature = grd_feature.view(N, C, H_k, W_k)

        # # only use kernel_edge+2shift_range in center as input
        # W_start = W_s // 2 - self.shift_range - kernel_W // 2
        # W_end = W_s // 2 + self.shift_range + kernel_W // 2
        # H_start = H_s // 2 - self.shift_range - kernel_H // 2
        # H_end = H_s // 2 + self.shift_range + kernel_H // 2
        # assert W_start >= 0 and W_end <= W_s, 'input of conv crop w error!!!'
        # assert H_start >= 0 and H_end <= H_s, 'input of conv crop h error!!!'
        # sat_feature = sat_feature[:, :, H_start:H_end, W_start:W_end]
        sat_feature = F.pad(sat_feature, (self.shift_range, self.shift_range, self.shift_range, self.shift_range))

        M, _, H_i, W_i = sat_feature.size()
        # if test_method != 2:
        sat_feature = F.normalize(sat_feature.reshape(M, -1))
        sat_feature = sat_feature.view(M, C, H_i, W_i)

        # corrilation to get similarity matrix
        in_feature = sat_feature.repeat(1, N, 1, 1)  # [M,C,H,W]->[M,N*C,H,W]
        correlation_matrix = F.conv2d(in_feature, grd_feature, groups=N)  # M, N, 2*shift_rang+1, 2*shift_rang+1)

        if test_method == 0 or test_method == 1:
            partical = F.avg_pool2d(sat_feature.pow(2), (H_k, W_k), stride=1,
                                    divisor_override=1)  # [M,C,2*shift_rang+1, 2*shift_rang+1]
            partical = torch.sum(partical, dim=1).unsqueeze(1)  # sum on C
            # partical = torch.maximum(partical, torch.ones_like(partical) * 1e-7) # for /0
            # assert torch.all(partical!=0.), 'have 0 in partical!!!'
            if test_method == 0:
                correlation_matrix /= torch.maximum(torch.sqrt(partical), torch.ones_like(partical) * 1e-12)
            else:
                correlation_matrix /= torch.maximum(partical, torch.ones_like(partical) * 1e-12)
            similarity_matrix = torch.amax(correlation_matrix, dim=(2, 3))  # M,N

            # if torch.max(similarity_matrix) > 1:
            #    print('>1,parical:',partical.cpu().detach(), correlation_matrix.cpu().detach())
        elif test_method == 2:
            W = correlation_matrix.size()[-1]
            max_index = torch.argmax(correlation_matrix.view(M, N, -1), dim=-1)  # M,N
            max_pos = torch.cat([(max_index // W).unsqueeze(-1), (max_index % W).unsqueeze(-1)], dim=-1)  # M,N,2

            # crop sat, and normalize
            in_feature = torch.tensor([], device=sat_feature.device)
            for i in range(N):
                sat_f_n = torch.tensor([], device=sat_feature.device)
                for j in range(M):
                    sat_f = sat_feature[j, :, max_pos[j, i, 0]:max_pos[j, i, 0] + H_k,
                            max_pos[j, i, 1]:max_pos[j, i, 1] + W_k]  # [C,H,W]
                    sat_f_n = torch.cat([sat_f_n, sat_f.unsqueeze(0)], dim=0)  # [M,C,H,W]
                in_feature = torch.cat([in_feature, sat_f_n.unsqueeze(1)], dim=1)  # [M,N,C,H,W]

            in_feature = F.normalize(in_feature.reshape(M * N, -1))
            in_feature = in_feature.view(M, N * C, H_k, W_k)

            grd_feature = F.normalize(grd_feature.reshape(N, -1))
            grd_feature = grd_feature.view(N, C, H_k, W_k)

            similarity_matrix = F.conv2d(in_feature, grd_feature, groups=N)  # M, N, 1,1)
            similarity_matrix = similarity_matrix.view(M, N)
        elif test_method == 3 or test_method == 4:
            max_values = torch.amax(correlation_matrix, dim=(2, 3), keepdim=True)
            max_values_mask = correlation_matrix == max_values

            partical = F.avg_pool2d(sat_feature.pow(2), (kernel_H, kernel_W), stride=1,
                                    divisor_override=1)  # [M,C,2*shift_rang+1, 2*shift_rang+1]
            partical = torch.sum(partical, dim=1).unsqueeze(1)  # sum on C

            if test_method == 3:
                max_values = correlation_matrix * max_values_mask / torch.maximum(torch.sqrt(partical),
                                                                                  torch.ones_like(partical) * 1e-12)
            else:
                max_values = correlation_matrix * max_values_mask / torch.maximum(partical,
                                                                                  torch.ones_like(partical) * 1e-12)
            similarity_matrix = torch.sum(max_values, dim=(2, 3))  # M,N

        # print('similarity_matric max&min:',torch.max(similarity_matrix).item(), torch.min(similarity_matrix).item() )
        return similarity_matrix


class partical_similarity_loss(nn.Module):
    """
    CVM
    """

    ### the value of margin is given according to the facenet
    def __init__(self, shift_range, loss_weight=10.0, test_method=2):
        super(partical_similarity_loss, self).__init__()
        self.similarity_function = partical_similarity(shift_range)
        self.loss_weight = loss_weight
        # self.use_corr = use_corr
        self.test_method = test_method
        self.shift_range = shift_range

    def forward(self, sat_feature, grd_feature):
        B = grd_feature.size()[0]

        if self.shift_range > 0:
            sim = self.similarity_function(grd_feature, sat_feature, self.test_method)
            dist_array = 2 - 2 * sim  # range: 2~0
        else:
            sat_feature = sat_feature.view(B, -1)
            grd_feature = grd_feature.view(B, -1)
            sat_feature = F.normalize(sat_feature)
            grd_feature = F.normalize(grd_feature)
            dist_array = 2 - 2 * sat_feature @ grd_feature.T
        pos_dist = torch.diagonal(dist_array)

        pair_n = B * (B - 1)

        # ground to satellite
        triplet_dist_g2s = pos_dist - dist_array
        triplet_dist_g2s = torch.log(1 + torch.exp(triplet_dist_g2s * self.loss_weight))
        # triplet_dist_g2s = triplet_dist_g2s - torch.diag(torch.diagonal(triplet_dist_g2s))
        # top_k_g2s, _ = torch.topk((triplet_dist_g2s.t()), B)
        # loss_g2s = torch.mean(top_k_g2s)
        loss_g2s = torch.sum(triplet_dist_g2s) / pair_n

        # satellite to ground
        triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
        triplet_dist_s2g = torch.log(1 + torch.exp(triplet_dist_s2g * self.loss_weight))
        # triplet_dist_s2g = triplet_dist_s2g - torch.diag(torch.diagonal(triplet_dist_s2g))
        # top_k_s2g, _ = torch.topk(triplet_dist_s2g, B)
        # loss_s2g = torch.mean(top_k_s2g)
        loss_s2g = torch.sum(triplet_dist_s2g) / pair_n

        loss = (loss_g2s + loss_s2g) / 2.0
        # loss = loss_g2s

        real_pos_dist_avg = pos_dist.mean()
        pred_pos_dist_avg = (torch.min(dist_array, dim=1)[0]).mean()
        # nega_dist_avg = dist_array.mean()

        return 10 * loss, real_pos_dist_avg, pred_pos_dist_avg



class similarity_uncertainty(nn.Module):
    def __init__(self, shift_range):
        super(similarity_uncertainty, self).__init__()
        self.shift_range = shift_range

    def forward(self, grd_feature, sat_feature, uncertainty=1, test_method=1):
        N, C, H_k, W_k = grd_feature.size()
        grd_feature = F.normalize(grd_feature.reshape(N, -1))
        grd_feature = grd_feature.view(N, C, H_k, W_k)

        sat_feature = F.pad(sat_feature, (self.shift_range, self.shift_range, self.shift_range, self.shift_range))

        M, _, H_i, W_i = sat_feature.size()
        sat_feature = F.normalize(sat_feature.reshape(M, -1))
        sat_feature = sat_feature.view(M, C, H_i, W_i)

        # corrilation to get similarity matrix
        in_feature = sat_feature.repeat(1, N, 1, 1)  # [M,C,H,W]->[M,N*C,H,W]
        correlation_matrix = F.conv2d(in_feature, grd_feature, groups=N)  # M, N, 2*shift_rang+1, 2*shift_rang+1)

        partical = F.avg_pool2d(sat_feature.pow(2), (H_k, W_k), stride=1,
                                divisor_override=1)  # [M,C,2*shift_rang+1, 2*shift_rang+1]
        partical = torch.sum(partical, dim=1).unsqueeze(1)  # sum on C

        correlation_matrix /= torch.maximum(torch.sqrt(partical) * uncertainty, torch.ones_like(partical) * 1e-12)

        similarity_matrix = torch.amax(correlation_matrix, dim=(2, 3))  # M,N

        # -------- Compute the relative shift between ground camera and satellite image center --------
        W = correlation_matrix.size()[-1]
        max_index = torch.argmax(correlation_matrix.view(M, N, -1), dim=-1)  # M,N
        max_pos = torch.cat([-((max_index // W).unsqueeze(-1) - W/2),
                             (max_index % W).unsqueeze(-1) - W/2], dim=-1)  # M,N,2
        shift_meters = max_pos * utils.get_meter_per_pixel() * utils.SatMap_original_sidelength /sat_feature.shape[-1]
        # [M, N, 2]
        return similarity_matrix, shift_meters


class loss_uncertainty(nn.Module):

    ### the value of margin is given according to the facenet
    def __init__(self, shift_range, loss_weight=10.0, test_method=2):
        super(loss_uncertainty, self).__init__()
        self.similarity_function = similarity_uncertainty(shift_range)
        self.loss_weight = loss_weight
        # self.use_corr = use_corr
        self.test_method = test_method
        self.shift_range = shift_range

    def forward(self, sat_feature, grd_feature, uncertainty=1):
        B = grd_feature.size()[0]

        if self.shift_range > 0:
            sim, _ = self.similarity_function(grd_feature, sat_feature, uncertainty, self.test_method)
            dist_array = 2 - 2 * sim  # range: 2~0
        else:
            sat_feature = sat_feature.view(B, -1)
            grd_feature = grd_feature.view(B, -1)
            sat_feature = F.normalize(sat_feature)
            grd_feature = F.normalize(grd_feature)
            dist_array = 2 - 2 * sat_feature @ grd_feature.T
        pos_dist = torch.diagonal(dist_array)

        pair_n = B * (B - 1)

        # ground to satellite
        triplet_dist_g2s = pos_dist - dist_array
        triplet_dist_g2s = torch.log(1 + torch.exp(triplet_dist_g2s * self.loss_weight))
        # triplet_dist_g2s = triplet_dist_g2s - torch.diag(torch.diagonal(triplet_dist_g2s))
        # top_k_g2s, _ = torch.topk((triplet_dist_g2s.t()), B)
        # loss_g2s = torch.mean(top_k_g2s)
        loss_g2s = torch.sum(triplet_dist_g2s) / pair_n

        # satellite to ground
        triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
        triplet_dist_s2g = torch.log(1 + torch.exp(triplet_dist_s2g * self.loss_weight))
        # triplet_dist_s2g = triplet_dist_s2g - torch.diag(torch.diagonal(triplet_dist_s2g))
        # top_k_s2g, _ = torch.topk(triplet_dist_s2g, B)
        # loss_s2g = torch.mean(top_k_s2g)
        loss_s2g = torch.sum(triplet_dist_s2g) / pair_n

        loss = (loss_g2s + loss_s2g) / 2.0
        # loss = loss_g2s

        real_pos_dist_avg = pos_dist.mean()
        pred_pos_dist_avg = (torch.min(dist_array, dim=1)[0]).mean()
        # nega_dist_avg = dist_array.mean()

        return 10 * loss, real_pos_dist_avg, pred_pos_dist_avg



class similarity_visualize_uncertainty(nn.Module):
    def __init__(self, shift_range):
        super(similarity_visualize_uncertainty, self).__init__()
        self.shift_range = shift_range

    def forward(self, grd_feature, sat_feature, uncertainty=1, test_method=1):
        N, C, H_k, W_k = grd_feature.size()
        grd_feature = F.normalize(grd_feature.reshape(N, -1))
        grd_feature = grd_feature.view(N, C, H_k, W_k)

        sat_feature = F.pad(sat_feature, (self.shift_range, self.shift_range, self.shift_range, self.shift_range))

        M, _, H_i, W_i = sat_feature.size()
        sat_feature = F.normalize(sat_feature.reshape(M, -1))
        sat_feature = sat_feature.view(M, C, H_i, W_i)

        # corrilation to get similarity matrix
        in_feature = sat_feature.repeat(1, N, 1, 1)  # [M,C,H,W]->[M,N*C,H,W]
        correlation_matrix = F.conv2d(in_feature, grd_feature, groups=N)  # M, N, 2*shift_rang+1, 2*shift_rang+1)

        partical = F.avg_pool2d(sat_feature.pow(2), (H_k, W_k), stride=1,
                                divisor_override=1)  # [M,C,2*shift_rang+1, 2*shift_rang+1]
        partical = torch.sum(partical, dim=1).unsqueeze(1)  # sum on C

        correlation_matrix /= torch.maximum(torch.sqrt(partical) * uncertainty, torch.ones_like(partical) * 1e-12)

        similarity_matrix = torch.amax(correlation_matrix, dim=(2, 3))  # M,N

        # # -------- Compute the relative shift between ground camera and satellite image center --------
        # W = correlation_matrix.size()[-1]
        # max_index = torch.argmax(correlation_matrix.view(M, N, -1), dim=-1)  # M,N
        # max_pos = torch.cat([-((max_index // W).unsqueeze(-1) - W/2),
        #                      (max_index % W).unsqueeze(-1) - W/2], dim=-1)  # M,N,2
        # shift_meters = max_pos * utils.get_meter_per_pixel() * utils.SatMap_original_sidelength /sat_feature.shape[-1]
        # # [M, N, 2]

        # --------- generate mask -------------
        max_index = torch.argmax(torch.diagonal(correlation_matrix, dim1=0, dim2=1).permute(2, 0, 1).view(M, -1),
                                 dim=-1)  # M,N
        max_pos = torch.cat([(max_index // correlation_matrix.shape[-1]).unsqueeze(-1),
                             (max_index % correlation_matrix.shape[-1]).unsqueeze(-1)], dim=-1)  # M,N,2
        mask = torch.zeros_like(sat_feature[:, 0:1, ...], device=sat_feature.device)
        grd_mask = torch.abs(torch.sum(grd_feature, dim=1)) > 1e-6
        for idx in range(M):
            mask[idx, 0, max_pos[idx, 0]: max_pos[idx, 0] + H_k, max_pos[idx, 1]: max_pos[idx, 1] + W_k] \
                = grd_mask[idx]

        return similarity_matrix, mask[:, :, self.shift_range: self.shift_range + H_k, self.shift_range: self.shift_range + H_k]



class similarity_for_visualize(nn.Module):
    def __init__(self, shift_range):
        super(similarity_for_visualize, self).__init__()
        self.shift_range = shift_range

    def forward(self, grd_feature, sat_feature, test_method=1):
        # test_method: 0:sqrt, 1:**2, 2:crop

        # use grd as kernel, sat map as inputs, convolution to get correlation
        M, C, H_s, W_s = sat_feature.size()

        _, _, H, W = grd_feature.size()
        kernel_H = min(H, H_s - 2 * self.shift_range)  # 24: 38 meters ahead and 38 meters behind
        kernel_W = min(W, W_s - 2 * self.shift_range)

        # only use kernel_edge in center as kernel, for efficient process time
        W_start = W // 2 - kernel_W // 2
        W_end = W // 2 + kernel_W // 2
        H_start = H // 2 - kernel_H // 2
        H_end = H // 2 + kernel_H // 2
        grd_feature = grd_feature[:, :, H_start:H_end, W_start:W_end]
        N, _, H_k, W_k = grd_feature.size()
        # if test_method != 2: #not crop_method: # normalize later
        grd_feature = F.normalize(grd_feature.reshape(N, -1))
        grd_feature = grd_feature.view(N, C, H_k, W_k)

        # only use kernel_edge+2shift_range in center as input
        W_start = W_s // 2 - self.shift_range - kernel_W // 2
        W_end = W_s // 2 + self.shift_range + kernel_W // 2
        H_start = H_s // 2 - self.shift_range - kernel_H // 2
        H_end = H_s // 2 + self.shift_range + kernel_H // 2
        assert W_start >= 0 and W_end <= W_s, 'input of conv crop w error!!!'
        assert H_start >= 0 and H_end <= H_s, 'input of conv crop h error!!!'
        sat_feature = sat_feature[:, :, H_start:H_end, W_start:W_end]
        _, _, H_i, W_i = sat_feature.size()
        # if test_method != 2:
        sat_feature = F.normalize(sat_feature.reshape(M, -1))
        sat_feature = sat_feature.view(M, C, H_i, W_i)

        # corrilation to get similarity matrix
        in_feature = sat_feature.repeat(1, N, 1, 1)  # [M,C,H,W]->[M,N*C,H,W]
        correlation_matrix = F.conv2d(in_feature, grd_feature, groups=N)  # M, N, 2*shift_rang+1, 2*shift_rang+1)

        if test_method != 2:
            partical = F.avg_pool2d(sat_feature.pow(2), (kernel_H, kernel_W), stride=1,
                                    divisor_override=1)  # [M,C,2*shift_rang+1, 2*shift_rang+1]
            partical = torch.sum(partical, dim=1).unsqueeze(1)  # sum on C
            # partical = torch.maximum(partical, torch.ones_like(partical) * 1e-7) # for /0
            # assert torch.all(partical!=0.), 'have 0 in partical!!!'
            if test_method == 0:
                correlation_matrix /= torch.maximum(torch.sqrt(partical), torch.ones_like(partical) * 1e-12)
            else:
                correlation_matrix /= torch.maximum(partical, torch.ones_like(partical) * 1e-12)
            similarity_matrix = torch.amax(correlation_matrix, dim=(2, 3))  # M,N

            # --------- generate mask -------------
            max_index = torch.argmax(torch.diagonal(correlation_matrix, dim1=0, dim2=1).permute(2, 0, 1).view(M, -1), dim=-1)  # M,N
            max_pos = torch.cat([(max_index // correlation_matrix.shape[-1]).unsqueeze(-1), (max_index % correlation_matrix.shape[-1]).unsqueeze(-1)], dim=-1)  # M,N,2
            mask = torch.zeros_like(sat_feature[:, 0:1, ...], device=sat_feature.device)
            grd_mask = torch.abs(torch.sum(grd_feature, dim=1)) > 1e-6
            for idx in range(M):
                mask[idx, 0, max_pos[idx, 0]: max_pos[idx, 0] + H_k, max_pos[idx, 1]: max_pos[idx, 1] + W_k] \
                    = grd_mask[idx]

        else:
            W = correlation_matrix.size()[-1]
            max_index = torch.argmax(correlation_matrix.view(M, N, -1), dim=-1)  # M,N
            max_pos = torch.cat([(max_index // W).unsqueeze(-1), (max_index % W).unsqueeze(-1)], dim=-1)  # M,N,2

            # crop sat, and normalize
            in_feature = torch.tensor([], device=sat_feature.device)
            for i in range(N):
                sat_f_n = torch.tensor([], device=sat_feature.device)
                for j in range(M):
                    sat_f = sat_feature[j, :, max_pos[j, i, 0]:max_pos[j, i, 0] + H_k,
                            max_pos[j, i, 1]:max_pos[j, i, 1] + W_k]  # [C,H,W]
                    sat_f_n = torch.cat([sat_f_n, sat_f.unsqueeze(0)], dim=0)  # [M,C,H,W]
                in_feature = torch.cat([in_feature, sat_f_n.unsqueeze(1)], dim=1)  # [M,N,C,H,W]

            in_feature = F.normalize(in_feature.reshape(M * N, -1))
            in_feature = in_feature.view(M, N * C, H_k, W_k)

            grd_feature = F.normalize(grd_feature.reshape(N, -1))
            grd_feature = grd_feature.view(N, C, H_k, W_k)

            similarity_matrix = F.conv2d(in_feature, grd_feature, groups=N)  # M, N, 1,1)
            similarity_matrix = similarity_matrix.view(M, N)

            # ---------- Generate mask --------------
            max_pos = torch.diagonal(max_pos, dim1=0, dim2=1).permute(2, 0, 1)  # M,2
            mask = torch.zeros_like(sat_feature[:, 0:1, ...], device=sat_feature.device)
            grd_mask = torch.abs(torch.sum(grd_feature, dim=1)) > 1e-6
            for idx in range(M):
                mask[idx, 0, max_pos[idx, 0]: max_pos[idx, 0] + H_k, max_pos[idx, 1]: max_pos[idx, 1] + W_k] \
                    = grd_mask[idx]

        # print('similarity_matric max&min:',torch.max(similarity_matrix).item(), torch.min(similarity_matrix).item() )
        return similarity_matrix, mask


class visualization_loss_uncertainty(nn.Module):

    ### the value of margin is given according to the facenet
    def __init__(self, shift_range, loss_weight=10.0, test_method=2):
        super(visualization_loss_uncertainty, self).__init__()
        self.similarity_function = similarity_visualize_uncertainty(shift_range)
        self.loss_weight = loss_weight
        # self.use_corr = use_corr
        self.test_method = test_method
        self.shift_range = shift_range

    def forward(self, sat_feature, grd_feature, uncertainty=1):

        sim, mask = self.similarity_function(grd_feature, sat_feature, uncertainty, self.test_method)
        dist_array = 2 - 2 * sim  # range: 2~0

        pos_dist = torch.diagonal(dist_array)

        return torch.mean(pos_dist), mask



class visualization_loss(nn.Module):
    """
    CVM
    """
    ### the value of margin is given according to the facenet
    def __init__(self, shift_range, loss_weight=10.0, test_method=2):
        super(visualization_loss, self).__init__()
        self.similarity_function = similarity_for_visualize(shift_range)
        self.loss_weight = loss_weight
        # self.use_corr = use_corr
        self.test_method = test_method
        self.shift_range = shift_range
        
    def forward(self, sat_feature, grd_feature):
        B = grd_feature.size()[0]
        
        if self.shift_range>0:
            sim, mask = self.similarity_function(grd_feature, sat_feature, self.test_method)
            dist_array =  2-2*sim # range: 2~0
        else:
            sat_feature = sat_feature.view(B,-1)
            grd_feature = grd_feature.view(B,-1)
            sat_feature = F.normalize(sat_feature) 
            grd_feature = F.normalize(grd_feature) 
            dist_array =  2 - 2*sat_feature@grd_feature.T
        pos_dist = torch.diagonal(dist_array)

        return torch.mean(pos_dist), mask

       
