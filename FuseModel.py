#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import transforms
import utils
import os

# from GRU1 import ElevationEsitimate,VisibilityEsitimate,VisibilityEsitimate2,GRUFuse
from VGG import VGG16

from ConvLSTM import VisibilityFusion, LSTMFusion, Conv3DFusion, Conv2DFusion
from Transformer import TransformerFusion


class uncertainty(nn.Module):
    def __init__(self, kernel=4, layer=8, shift_range=3):
        super(uncertainty, self).__init__()

        self.convs = nn.ModuleList()

        for idx in range(layer - 1):
            self.convs.extend([
                nn.ReLU(),
                nn.Conv2d(4, 4, kernel_size=(kernel + 1, kernel + 1), stride=(1, 1), padding=(0, 0)),
            ])
        self.convs.extend([
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=(kernel, kernel), stride=(1, 1), padding=(0, 0)),
            nn.Sigmoid(),
        ])
        self.shift_range = shift_range

    def forward(self, x):
        y = F.pad(x, (self.shift_range, self.shift_range, self.shift_range, self.shift_range))
        for layer in self.convs:
            y = layer(y)
        return y

class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
        self.encs = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
        )

        self.decs0 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.decs1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.decs2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x):
        # x.shape = [B, C, H, W]
        B, C, H, W = x.shape

        x0 = self.encs(x)
        x0 = x0.reshape(B, C*4, H // 2, W // 8)
        y0 = self.decs0(x0)
        y1 = F.interpolate(y0, (H, W // 4))
        y1 = self.decs1(y1)
        y2 = F.interpolate(y1, (H * 2, W // 2))
        y2 = self.decs2(y2)

        return y2


class FuseModel(nn.Module):
    def __init__(self, debug_flag=0, sequence=1, stereo=False, feature_win=32, sim=0,
                 fuse_method='vis_Conv2D', seq_order=0, shift_range=3, proj='Geometry'):  # device='cuda:0',
        '''
        fuse_method: vis_Conv2D, vis_LSTM, vis_Con3D, fuse_LSTM, fuse_Transformer
        proj: Geometry, Unet, Reshape
        '''
        super(FuseModel, self).__init__()
        self.debug_flag = debug_flag

        self.sequence = sequence
        self.stereo = stereo

        self.feature_win = feature_win
        # self.height_planes = height_planes
        self.fuse_method = fuse_method
        # self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.proj = proj

        out_c = 16
        self.SatFeatureNet = VGG16(num_classes=out_c, win_size=feature_win)  # SiamFCANet(num_classes = out_c)
        self.GrdFeatureNet = VGG16(num_classes=out_c, win_size=feature_win)  # SiamFCANet(num_classes = out_c)

        if self.proj == 'Unet':
            self.ProjNet = unet()

        self.stereo = stereo
        self.UncertaintyNet = uncertainty(kernel=4, layer=8, shift_range=shift_range)

        if sim == 0:
            input_dim = out_c  # + 2
        else:
            input_dim = out_c + 2
        hidden_dim = input_dim // 2

        self.sim = sim
        # self.height_sample = height_sample

        if fuse_method.startswith('vis_'):  # 'LSTM3D_2, LSTM_conv_2, LSTM_LSTM_2, conv_LSTM_2, conv_conv_2'

            self.FuseNet = VisibilityFusion(input_dim, hidden_dim, kernel_size=(3, 3), num_layers=2, seq_output_dim=hidden_dim,
                                     bias=True, seq_order=seq_order, seq_fuse=fuse_method.split('_')[1])
        elif fuse_method == 'fuse_LSTM':
            self.FuseNet = LSTMFusion(seq_num=sequence, seq_input_dim=input_dim, hidden_dim=hidden_dim,
                                      kernel_size=(3, 3), num_layers=2, seq_output_dim=hidden_dim,
                                      bias=True, seq_order=seq_order)

        elif fuse_method == 'fuse_Conv3D':
            self.FuseNet = Conv3DFusion(seq_num=sequence, seq_input_dim=input_dim, hidden_dim=hidden_dim,
                                      kernel_size=(3, 3), num_layers=2, seq_output_dim=hidden_dim,
                                      bias=True, seq_order=seq_order)
        elif fuse_method == 'fuse_Conv2D':
            self.FuseNet = Conv2DFusion(seq_num=sequence, seq_input_dim=input_dim, hidden_dim=hidden_dim,
                                      kernel_size=(3, 3), num_layers=2, seq_output_dim=hidden_dim,
                                      bias=True, seq_order=seq_order)
        elif fuse_method == 'fuse_Transformer':
            self.FuseNet = TransformerFusion(seq=sequence, n_embd=input_dim, n_head=2, 
                                             n_layers=2)
        
        self.fuse_method = fuse_method

        Grd_Downch_Conv1 = nn.Conv2d(out_c, 16, (3, 3), padding=1, bias=False)
        Grd_Downch_Conv2 = nn.Conv2d(16, 4, (3, 3), padding=1, bias=False)

        Sat_Downch_Conv1 = nn.Conv2d(out_c, 16, (3, 3), padding=1, bias=False)
        Sat_Downch_Conv2 = nn.Conv2d(16, 4, (3, 3), padding=1, bias=False)

        self.SatDownch = nn.Sequential(Sat_Downch_Conv1, nn.ReLU(), Sat_Downch_Conv2)
        self.GrdDownch = nn.Sequential(Grd_Downch_Conv1, nn.ReLU(), Grd_Downch_Conv2)

        torch.autograd.set_detect_anomaly(True)
        # Running the forward pass with detection enabled will allow the backward pass to print the traceback of the forward operation that created the failing backward function.
        # Any backward computation that generate “nan” value will raise an error.

    def get_warp_sat2real(self, satmap_sidelength, min_height=0, max_height=8):
        # satellite: u:east , v:south from bottomleft and u_center: east; v_center: north from center
        # realword: X: south, Y:down, Z: east   origin is set to the ground plane

        # meshgrid the sat pannel
        i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
        ii, jj = torch.meshgrid(i, j)  # i:h,j:w

        # uv is coordinate from top/left, v: south, u:east
        uv = torch.stack([jj, ii], dim=-1).float()  # shape = [satmap_sidelength, satmap_sidelength, 2]

        # sat map from top/left to center coordinate
        u0 = v0 = satmap_sidelength // 2
        uv_center = uv - torch.tensor(
            [u0, v0]).cuda()  # .to(self.device) # shape = [satmap_sidelength, satmap_sidelength, 2]

        # affine matrix: scale*R
        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / self.feature_win
        R = torch.tensor([[0, 1], [1, 0]]).float().cuda()  # to(self.device) # u_center->z, v_center->x
        Aff_sat2real = meter_per_pixel * R  # shape = [2,2]

        # Trans matrix from sat to realword
        XZ = torch.einsum('ij, hwj -> hwi', Aff_sat2real,
                          uv_center)  # shape = [satmap_sidelength, satmap_sidelength, 2]

        # if self.height_sample == 'uniform':
        #     Y = torch.linspace(min_height, max_height, self.height_planes).cuda()
        # elif self.height_sample == 'inverse':
        #     delta = 1
        #     max_height += delta
        #     min_height += delta
        #     Y = (max_height - 1 / torch.linspace(1 / max_height, 1 / min_height, self.height_planes)).cuda()
        #
        # Y = Y.view(-1, 1, 1, 1)
        # Y = Y.expand(-1, satmap_sidelength, satmap_sidelength, -1)  # [height, sidelength,sidelength,1]
        # self.Y = Y
        Y = torch.zeros(1, satmap_sidelength, satmap_sidelength, 1, dtype=XZ.dtype, device=XZ.device)

        XZ = torch.unsqueeze(XZ, 0)  # [1,sidelength,sidelength,2]
        # XZ = XZ.expand(self.height_planes, -1, -1, -1)  # [height,sidelength,sidelength,2]
        ones = torch.ones_like(Y)
        sat2realwap = torch.cat([XZ[:, :, :, :1], Y, XZ[:, :, :, 1:], ones], dim=-1)  # [1,sidelength,sidelength,4]

        return sat2realwap

    def seq_warp_real2camera(self, XYZ_1, heading, camera_k, shift):
        # realword: X: northsouth, Y:down, Z: east
        # camera: u:south, v: down from center (when heading east, need rotate heading angle)
        # XYZ_1:[height=1, H,W,4], heading:[B,S], camera_k:[B,3,3], shift:[B,S,2]
        # E=1 in this function
        B, S = heading.size()

        # R = utils.Rotation_y(-heading) # shape = [B,S,3,3]
        cos = torch.cos(-heading).unsqueeze(-1)
        sin = torch.sin(-heading).unsqueeze(-1)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B,S,9]
        R = R.view(B, S, 3, 3)  # shape = [B,S,3,3]

        camera_height = utils.get_camera_height()
        # camera offset, shift[0]:east,Z, shift[1]:north,X
        height = camera_height * torch.ones_like(shift[:, :, :1])
        T = torch.cat([shift[:, :, 1:], height, -shift[:, :, :1]], dim=-1)  # shape = [B,S,3]
        T = torch.unsqueeze(T, dim=-1)  # shape = [B,S,3,1]
        T = torch.einsum('bsij, bsjk -> bsik', R, T)

        # P = K[R|T]
        camera_k[:, :1,
        :] *= self.feature_win * 2 / 1024  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] *= self.feature_win / 2 / 256
        P = torch.einsum('bij, bsjk -> bsik', camera_k, torch.cat([R, T], dim=-1)).float()  # shape = [B,S,3,4]

        uv_1 = torch.einsum('bsij, ehwj -> bsehwi', P, XYZ_1)  # shape = [B,S,E,H, W,3]
        # only need view in front of camera ,Epsilon = 1e-6
        uv_1_last = torch.maximum(uv_1[:, :, :, :, :, 2:], torch.ones_like(uv_1[:, :, :, :, :, 2:]) * 1e-6)
        uv = uv_1[:, :, :, :, :, :2] / uv_1_last  # shape = [B,S, E, H, W,2]

        return uv

    def project_seq_grd_to_map(self, grd_f, shift, heading, camera_k, satmap_sidelength):
        # inputs:
        #   grd_f: ground features: B,S,C,H,W
        #   shift: B, S, 3
        #   heading: heading angle: B,S
        #   camera_k: 3*3 K matrix of left color camera : B*3*3
        # return:
        #   grd_f_trans: B,S,E,C,satmap_sidelength,satmap_sidelength

        B, S, C, H, W = grd_f.size()

        # get warp matrix
        XYZ_1 = self.get_warp_sat2real(satmap_sidelength)  # [height, sidelength,sidelength,4]
        # get shift between satellite and camera
        # shift_left, shift_right = self.get_seq_shift_meter(loc_left, loc_right) # [B,S,2]
        uv = self.seq_warp_real2camera(XYZ_1, heading, camera_k, shift)  # [B, S, E, H, W,2]

        # normalize to [-1, 1] for F.grid_sample
        uv_center = uv - torch.tensor([W // 2, H // 2]).cuda()  # shape = [B, S, E, H, W,2]
        # u:north, v: up from center to -1,-1 top left, 1,1 buttom rightVisibility_elevation_fuse
        scale = torch.tensor([W // 2, H // 2]).cuda()
        uv_center /= scale

        # expand grd_f to [B, S, E, C, H, W]
        E = uv.size()[2]
        grd_f = grd_f.unsqueeze(2).repeat(1, 1, E, 1, 1, 1)
        grd_f_trans = F.grid_sample(grd_f.reshape(-1, C, H, W),
                                    uv_center.reshape(-1, satmap_sidelength, satmap_sidelength, 2), mode='bilinear',
                                    padding_mode='zeros')  # [B*S*E,C,sidelength,sidelength]
        grd_f_trans = grd_f_trans.view(B, S, E, C, satmap_sidelength, satmap_sidelength)

        return grd_f_trans

    def Merge_multi_grd2sat(self, shift_left, shift_right, grd_f_left, grd_f_right, left_camera_k, right_camera_k,
                            heading, satmap_sidelength):
        # grd_img_left,grd_img_right: [B,S,C,H,W]
        # coarse_loc: [B,S,3] heading: [B,S]
        B, S, C, _, _ = grd_f_left.size()

        if grd_f_right != None and 0 not in grd_f_right.shape:
            grd_tran_left = self.project_seq_grd_to_map \
                (grd_f_left, shift_left, heading, left_camera_k, satmap_sidelength)  # [B,S,E,C,H,W]

            grd_tran_right = self.project_seq_grd_to_map \
                (grd_f_right, shift_right, heading, right_camera_k, satmap_sidelength)  # [B,S, E,C,H,W]
            E = grd_tran_left.size()[2]
            grd_tran_final = torch.cat([grd_tran_left.unsqueeze(2), grd_tran_right.unsqueeze(2)], dim=2).view(B, S * 2,
                                                                                                              E, C,
                                                                                                              satmap_sidelength,
                                                                                                              satmap_sidelength)  # [B,2S,E,C,H,W]
            # grd_tran_final = torch.cat([grd_tran_left, grd_tran_right], dim=1)  #[B,2S,E,C,H,W]

        else:
            # process together
            grd_tran_left = self.project_seq_grd_to_map(grd_f_left, shift_left, heading, left_camera_k,
                                                        satmap_sidelength)  # [B,S,E,C,H,W]
            grd_tran_final = grd_tran_left

        if self.debug_flag:
            out_dir = './visualize/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            for idx in range(B):

                for seq_idx in range(S):
                    grd_img = transforms.functional.to_pil_image(grd_tran_left[idx, seq_idx, 0], mode='RGB')
                    grd_img.save(os.path.join(out_dir, 'grd_left_trans_B' + str(idx) + '_S' + str(seq_idx) + '.png'))

        return grd_tran_final

    def SequenceFusion(self, grd_feature, attn_pdrop=0.5, resid_pdrop=0.5, pe_pdrop=0.5):
        # input: grd_feature:[B,S,E,C,H,W]
        # output: grd_feature:[B,C,H,W]

        B, S, E, C, H, W = grd_feature.size()
        grd_feature = grd_feature[:, :, 0, :, :, :]

        if self.sim:
            similarity = torch.einsum('bemchw, becnhw->bemnhw', grd_feature.permute(0, 2, 1, 3, 4, 5),
                                      grd_feature.permute(0, 2, 3, 1, 4, 5))
            similarity = torch.mean(similarity, dim=3, keepdim=True)  # [B, E, S, 1, H, W]
            similarity_mean = torch.mean(similarity, dim=2, keepdim=True).repeat(1, 1, S, 1, 1, 1)
            # [B, E, S, 1, H, W]
            grd_feature = torch.cat([grd_feature, similarity.transpose(1, 2), similarity_mean.transpose(1, 2)], dim=3)
            # [B, S, E, C+2, H, W]

        if self.fuse_method == 'fuse_Transformer':
            x, att = self.FuseNet(grd_feature, attn_pdrop, resid_pdrop, pe_pdrop)
            fuse_feature = torch.mean(x, dim=1)



        else:
            fuse_feature = self.FuseNet(grd_feature)

        return fuse_feature  # [B,C,H,W]

    def forward(self, sat_map, left_camera_k, right_camera_k, grd_img_left, grd_img_right, loc_shift_left,
                loc_shift_right, heading, attn_pdrop=0, resid_pdrop=0, pe_pdrop=0):

        # sat_map, left_camera_k, right_camera_k, grd_img_left, grd_img_right, \
        # loc_shift_left, loc_shift_right, heading = x
        # grd_img_left, grd_img_right: [B,S,C,H,W]
        # loc_shift_left: [B,S,2] heading: [B,S]
        # left_camera_k,right_camera_k [B,3,3]
        # sat_map: [B,C,H,W]  loc_shift_right: [B, S, 2]

        H_s = self.feature_win
        sat_feature = None
        uncertainty = None
        if sat_map != None:
            sat_feature = self.SatFeatureNet(sat_map)
            sat_feature = self.SatDownch(sat_feature)

            B, C, H_s, W_s = sat_feature.size()

            if torch.max(sat_feature) - torch.min(sat_feature) < 1e-11:
                print('sat_features_max&min:', torch.max(sat_feature).item(), torch.min(sat_feature).item())
            # chek feature not all same
            assert torch.max(sat_feature) - torch.min(sat_feature) >= 1e-11, 'sat_feature all the same!!!'
            uncertainty = self.UncertaintyNet(sat_feature)

        # [B,S,E,C,H,W]
        if grd_img_left != None:
            B, S, C_in, H_in, W_in = grd_img_left.size()
            grd_feature_l = self.GrdFeatureNet(grd_img_left.view(-1, C_in, H_in, W_in))
            _, C, H_g, W_g = grd_feature_l.size()
            grd_feature_l = grd_feature_l.view(B, S, C, H_g, W_g)
            if self.stereo:
                grd_feature_r = self.GrdFeatureNet(grd_img_right.view(-1, C_in, H_in, W_in))
                grd_feature_r = grd_feature_r.view(B, S, C, H_g, W_g)
            else:
                grd_feature_r = None

            assert torch.max(grd_feature_l) - torch.min(grd_feature_l) >= 1e-11, 'grd_feature all the same!!!'

        # projection ground features to head view features
        grd_feature = None
        if grd_img_left != None:

            if self.debug_flag:
                grd_feature = self.Merge_multi_grd2sat \
                    (loc_shift_left, loc_shift_right, grd_img_left, grd_img_right, left_camera_k, right_camera_k,
                     heading, 512)  # [B,S,E,C,H,W]
            else:
                if self.proj == 'Geometry':
                    grd_feature = self.Merge_multi_grd2sat \
                        (loc_shift_left, loc_shift_right, grd_feature_l, grd_feature_r, left_camera_k, right_camera_k,
                         heading, H_s)  # [B,S,E,C,H,W]
                elif self.proj == 'Unet':
                    grd_feature = self.ProjNet(grd_feature_l.reshape(B * S, C, H_g, W_g))
                    grd_feature = grd_feature.reshape(B, S, 1, C, H_g * 2, W_g // 2)
                elif self.proj == 'Reshape':
                    grd_feature = grd_feature_l.reshape(B, S, C, H_g * 2, W_g // 2).unsqueeze(dim=2)

            grd_feature = self.SequenceFusion(grd_feature, attn_pdrop, resid_pdrop, pe_pdrop)  #[B,C,H,W]

            grd_feature = self.GrdDownch(grd_feature)



        return grd_feature, sat_feature, uncertainty



