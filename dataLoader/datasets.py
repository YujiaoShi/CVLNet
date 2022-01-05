import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

import torch
import pandas as pd
import utils

root_dir = '/media/yujiao/6TB/dataset/Kitti1' # '../../data/Kitti' # '../Data' #'..\\Data' #

test_csv_file_name = 'test.csv'
ignore_csv_file_name = 'ignore.csv'
satmap_dir = 'satmap'
grdimage_dir = 'raw_data'
left_color_camera_dir = 'image_02/data'  # 'image_02\\data' #
right_color_camera_dir = 'image_03/data'  # 'image_03\\data' #
oxts_dir = 'oxts/data'  # 'oxts\\data' #

GrdImg_H = 256  # 256 # original: 375 #224, 256
GrdImg_W = 1024  # 1024 # original:1242 #1248, 1024
GrdOriImg_H = 375
GrdOriImg_W = 1242
num_thread_workers = 1

# train_file = './dataLoader/train_files.txt'
train_file = './dataLoader/train_files_with_sat_GPS.txt'
test_file = './dataLoader/test2_files_with_sat_GPS.txt'
val_file = './dataLoader/test1_files_with_sat_GPS.txt'

semantic_dir = 'semantics'

class SatGrdDataset(Dataset):
    def __init__(self, root, file_name, stereo=False, sequence=False,
                 transform=None, use_polar_sat=0, use_project_grd=0, use_semantic=0):
        self.root = root
        self.stereo = stereo
        self.sequence = sequence

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.use_polar_sat = use_polar_sat

        if use_project_grd:
            self.pro_grdimage_dir = 'project_grd'
        else:
            self.pro_grdimage_dir = 'raw_data'

        self.satmap_dir = satmap_dir

        if use_polar_sat:
            self.satmap_dir = 'satmap_polar/train_10mgap'
        else:
            self.satmap_dir += '/train_10mgap'


        with open(file_name, 'r') as f:
            file_name = f.readlines()

        # self.file_name = [file[:-1] for file in file_name]

        self.semantic_dir = semantic_dir

        self.file_name = []
        for file in file_name:
            new_file = os.path.join(root, semantic_dir, '2011' + file.strip().split(' ')[0].split('/2011')[1])
            if not os.path.exists(new_file):
                print('File not exists: ', new_file)
                continue
            else:
                self.file_name.append(file.strip())

        self.use_semantic = use_semantic

        return

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        file_name, sat_lat, sat_lon = self.file_name[idx].split(' ')
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        sat_x, sat_y = utils.gps2utm(float(sat_lat), float(sat_lon))
        loc_sat = torch.tensor(np.array([sat_x, sat_y]))

        # =================== read file names within one sequence =====================
        sequence_list = []
        if self.sequence > 1:
            # need get sequence count files
            sequence_count = self.sequence

            # get sequence count files in drive_dir in before, if not enough, get after
            sequence_list.append(file_name)
            tar_image_no = int(image_no.split('.')[0])
            while len(sequence_list) < sequence_count:
                tar_image_no = tar_image_no - self.skip_in_seq - 1

                # create name of
                tar_img_no = '%010d' % (tar_image_no) + '.png'
                tar_file_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, right_color_camera_dir, tar_img_no)
                if os.path.exists(tar_file_name):
                    sequence_list.append(drive_dir + tar_img_no)
                else:
                    print('error, no enough sequence images in drive_dir:', drive_dir, len(sequence_list))
                    break
        else:
            sequence_list.append(file_name)

        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # left color camera k matrix
                if 'P_rect_02' in line:
                    # get 3*3 matrix from P_rect_**:
                    items = line.split(':')
                    valus = items[1].strip().split(' ')
                    fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                    cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                    fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                    cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                    left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    left_camera_k = torch.from_numpy(np.asarray(left_camera_k))
                    if not self.stereo:
                        break

                if self.stereo:
                    # right color camera K matrix
                    if 'P_rect_03' in line:
                        # get 3*3 matrix from P_rect_**:
                        items = line.split(':')
                        valus = items[1].strip().split(' ')
                        fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                        cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                        fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                        cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                        right_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                        right_camera_k = torch.from_numpy(np.asarray(right_camera_k))
                        break
                else:
                    right_camera_k = torch.tensor([])

        # =================== read satellite map ===================================
        file_name = sequence_list[0]
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        try:
            sat_map = Image.open(SatMap_name, 'r').convert('RGB')
        except:
            print('Read Fail: ', SatMap_name)

        # with Image.open(SatMap_name, 'r') as SatMap:
        #     sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        grd_right_imgs = torch.tensor([])

        loc_left_array = torch.tensor([])
        loc_right_array = torch.tensor([])

        heading_array = torch.tensor([])
        latlon_array = torch.tensor([])


        # =================== read locations at the first time step within a sequence ================
        #                         (note: not the one that should be located)
        file_last = sequence_list[-1]
        image_no_last = file_last[38:]

        # oxt: such as 0000000000.txt
        oxts_file_name_last = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no_last.lower().replace('.png', '.txt'))
        with open(oxts_file_name_last, 'r') as f:
            content = f.readline().split(' ')
        lat0 = float(content[0])

        # =================== read and compute relative locations within a sequence ==========
        # Although the GPS location is used here, our network only takes the relative shifts as input.
        # This confirms to the practical setting: we know the relative camera poses, and aims to localize the absolute camera locations.
        for i in range(len(sequence_list)):
            file_name = sequence_list[i]
            image_no = file_name[38:]

            # oxt: such as 0000000000.txt
            oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                          image_no.lower().replace('.png', '.txt'))
            with open(oxts_file_name, 'r') as f:
                content = f.readline().split(' ')

                # get heading
                heading = float(content[5])

                # get location
                # utm_x, utm_y = utils.gps2utm(float(content[0]), float(content[1]), lat0)  # location of the GPS device
                utm_x, utm_y = utils.gps2utm(float(content[0]), float(content[1]))
                delta_left_x, delta_left_y = utils.get_camera_gps_shift_left(
                    heading)  # delta x and delta y between the GPS and the left camera device
                delta_right_x, delta_right_y = utils.get_camera_gps_shift_right(
                    heading)  # delta x and delta y between the GPS and the left camera device

                left_x = utm_x + delta_left_x
                left_y = utm_y + delta_left_y

                right_x = utm_x + delta_right_x
                right_y = utm_y + delta_right_y

                loc_left = torch.from_numpy(np.asarray([left_x, left_y]))
                loc_right = torch.from_numpy(np.asarray([right_x, right_y]))
                heading = torch.from_numpy(np.asarray(heading))
                latlon = torch.from_numpy(np.asarray([float(content[0]), float(content[1])]))

                # ground images, left color camera
                left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                             image_no.lower())
                left_semantic_name = os.path.join(self.root, self.semantic_dir, '2011' + file_name.strip().split('/2011')[1])

                try:
                    GrdImg = Image.open(left_img_name, 'r')
                    grd_img_left = GrdImg.convert('RGB')


                    if self.use_semantic:
                        grd_semantic_left = np.array(Image.open(left_semantic_name, 'r'))
                        mask = (grd_semantic_left == 97) | (grd_semantic_left==98) | (grd_semantic_left==100) | (grd_semantic_left==101) | (grd_semantic_left==96)

                        grd_img_left = Image.fromarray(mask[..., None] * grd_img_left)

                    if self.grdimage_transform is not None:
                        grd_img_left = self.grdimage_transform(grd_img_left)

                except:
                    print('Error: ', left_img_name)

                # with Image.open(left_img_name, 'r') as GrdImg:
                #     grd_img_left = GrdImg.convert('RGB')
                #     if self.grdimage_transform is not None:
                #         grd_img_left = self.grdimage_transform(grd_img_left)

                if self.stereo:
                    # right color camera
                    right_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir,
                                                  right_color_camera_dir, image_no.lower())
                    with Image.open(right_img_name, 'r') as RightImg:
                        grd_img_right = RightImg.convert('RGB')
                        if self.grdimage_transform is not None:
                            grd_img_right = self.grdimage_transform(grd_img_right)

                #  add to tensor array
                grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)
                loc_left_array = torch.cat([loc_left_array, loc_left.unsqueeze(0)], dim=0)
                loc_right_array = torch.cat([loc_right_array, loc_right.unsqueeze(0)], dim=0)
                heading_array = torch.cat([heading_array, heading.unsqueeze(0)], dim=0)

                latlon_array = torch.cat([latlon_array, latlon.unsqueeze(0)], dim=0)
                # location (absolute) of the GPS device. It is given for evaluation.

                if self.stereo:
                    grd_right_imgs = torch.cat([grd_right_imgs, grd_img_right.unsqueeze(0)], dim=0)

        loc_shift_left = loc_left_array - loc_left_array[0:1, :]  # [N, 2], relative to the query location
        loc_shift_right = loc_right_array - loc_left_array[0:1, :]  # [N, 2], relative to the query location


        if "polar" not in self.satmap_dir:   # why shift satellite map?
            # x, y = utils.get_camera_gps_shift_left(heading_array[0].item())  # shift <1.4m
            # meter_per_pixel = utils.get_meter_per_pixel(scale=1)
            # shift_xy = (np.array([x, y]) / meter_per_pixel).astype(np.int32)

            # crop out the central region
            SatMap_sidelength = utils.get_original_satmap_sidelength()
            width, height = sat_map.size
            centroid_x = width // 2 #+ shift_xy[0]
            centroid_y = height // 2 #- shift_xy[1]
            left = centroid_x - SatMap_sidelength / 2
            top = centroid_y - SatMap_sidelength / 2

            # crop
            sat_map = sat_map.crop((left, top, left + SatMap_sidelength, top + SatMap_sidelength))

        else:
            sat_map = sat_map.resize((1024, 256))

        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        return sat_map, left_camera_k, right_camera_k, grd_left_imgs, grd_right_imgs, \
               loc_shift_left, loc_shift_right, heading_array, loc_left_array, loc_sat, file_name


class GrdDataset(Dataset):
    def __init__(self, root, stereo=False, sequence=False,
                 transform=None, use_project_grd=0, use_semantic=0):
        self.root = root
        self.stereo = stereo
        self.sequence = sequence

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            # self.satmap_transform = transform[0]
            self.grdimage_transform = transform

        if use_project_grd:
            self.pro_grdimage_dir = 'project_grd'
        else:
            self.pro_grdimage_dir = 'raw_data'

        with open(test_file, 'r') as f:
            file_name = f.readlines()

        # self.file_name = [file[:-1] for file in file_name]
        self.semantic_dir = semantic_dir

        self.file_name = []
        for file in file_name:
            new_file = os.path.join(root, semantic_dir, '2011' + file.strip().split(' ')[0].split('/2011')[1])
            if not os.path.exists(new_file):
                print('File not exists: ', new_file)
                continue
            else:
                self.file_name.append(file.strip())

        self.use_semantic = use_semantic

        return

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name
        file_name = self.file_name[idx].strip().split(' ')[0]
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        # =================== read file names within one sequence =====================
        sequence_list = []
        if self.sequence > 1:
            # need get sequence count files
            sequence_count = self.sequence

            # get sequence count files in drive_dir in before, if not enough, get after
            sequence_list.append(file_name)
            tar_image_no = int(image_no.split('.')[0])
            while len(sequence_list) < sequence_count:
                tar_image_no = tar_image_no - self.skip_in_seq - 1

                # create name of
                tar_img_no = '%010d' % (tar_image_no) + '.png'
                tar_file_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, right_color_camera_dir,
                                             tar_img_no)
                if os.path.exists(tar_file_name):
                    sequence_list.append(drive_dir + tar_img_no)
                else:
                    print('error, no enough sequence images in drive_dir:', drive_dir, len(sequence_list))
                    break
        else:
            sequence_list.append(file_name)

        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # left color camera k matrix
                if 'P_rect_02' in line:
                    # get 3*3 matrix from P_rect_**:
                    items = line.split(':')
                    valus = items[1].strip().split(' ')
                    fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                    cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                    fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                    cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                    left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    left_camera_k = torch.from_numpy(np.asarray(left_camera_k))
                    if not self.stereo:
                        break

                if self.stereo:
                    # right color camera K matrix
                    if 'P_rect_03' in line:
                        # get 3*3 matrix from P_rect_**:
                        items = line.split(':')
                        valus = items[1].strip().split(' ')
                        fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                        cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                        fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                        cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                        right_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                        right_camera_k = torch.from_numpy(np.asarray(right_camera_k))
                        break
                else:
                    right_camera_k = torch.tensor([])

        # =================== read satellite map ===================================
        # file_name = sequence_list[0]
        # SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        # with Image.open(SatMap_name, 'r') as SatMap:
        #     sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        grd_right_imgs = torch.tensor([])

        loc_left_array = torch.tensor([])
        loc_right_array = torch.tensor([])

        heading_array = torch.tensor([])
        latlon_array = torch.tensor([])

        # =================== read locations at the first time step within a sequence ================
        #                         (note: not the one that should be located)
        file_last = sequence_list[-1]
        image_no_last = file_last[38:]

        # oxt: such as 0000000000.txt
        oxts_file_name_last = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                           image_no_last.lower().replace('.png', '.txt'))
        with open(oxts_file_name_last, 'r') as f:
            content = f.readline().split(' ')
        lat0 = float(content[0])

        # =================== read and compute relative locations within a sequence ==========
        # Although the GPS location is used here, our network only takes the relative shifts as input.
        # This confirms to the practical setting: we know the relative camera poses, and aims to localize the absolute camera locations.
        for i in range(len(sequence_list)):
            file_name = sequence_list[i]
            image_no = file_name[38:]

            # oxt: such as 0000000000.txt
            oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                          image_no.lower().replace('.png', '.txt'))
            with open(oxts_file_name, 'r') as f:
                content = f.readline().split(' ')

                # get heading
                heading = float(content[5])

                # get location
                # utm_x, utm_y = utils.gps2utm(float(content[0]), float(content[1]), lat0)  # location of the GPS device
                utm_x, utm_y = utils.gps2utm(float(content[0]), float(content[1]))
                delta_left_x, delta_left_y = utils.get_camera_gps_shift_left(
                    heading)  # delta x and delta y between the GPS and the left camera device
                delta_right_x, delta_right_y = utils.get_camera_gps_shift_right(
                    heading)  # delta x and delta y between the GPS and the left camera device

                left_x = utm_x + delta_left_x
                left_y = utm_y + delta_left_y

                right_x = utm_x + delta_right_x
                right_y = utm_y + delta_right_y

                loc_left = torch.from_numpy(np.asarray([left_x, left_y]))
                loc_right = torch.from_numpy(np.asarray([right_x, right_y]))
                heading = torch.from_numpy(np.asarray(heading))
                latlon = torch.from_numpy(np.asarray([float(content[0]), float(content[1])]))

                # ground images, left color camera
                left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                             image_no.lower())

                left_semantic_name = os.path.join(self.root, self.semantic_dir,
                                                  '2011' + file_name.strip().split('/2011')[1])

                try:
                    GrdImg = Image.open(left_img_name, 'r')
                    grd_img_left = GrdImg.convert('RGB')

                    if self.use_semantic:
                        grd_semantic_left = np.array(Image.open(left_semantic_name, 'r'))
                        mask = (grd_semantic_left == 97) | (grd_semantic_left == 98) | (grd_semantic_left == 100) | (
                                    grd_semantic_left == 101) | (grd_semantic_left == 96)

                        grd_img_left = Image.fromarray(mask[..., None] * grd_img_left)

                    if self.grdimage_transform is not None:
                        grd_img_left = self.grdimage_transform(grd_img_left)
                except:
                    print('Error: ', left_img_name)


                # with Image.open(left_img_name, 'r') as GrdImg:
                #     grd_img_left = GrdImg.convert('RGB')
                #     if self.grdimage_transform is not None:
                #         grd_img_left = self.grdimage_transform(grd_img_left)

                if self.stereo:
                    # right color camera
                    right_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir,
                                                  right_color_camera_dir, image_no.lower())
                    with Image.open(right_img_name, 'r') as RightImg:
                        grd_img_right = RightImg.convert('RGB')
                        if self.grdimage_transform is not None:
                            grd_img_right = self.grdimage_transform(grd_img_right)

                #  add to tensor array
                grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)
                loc_left_array = torch.cat([loc_left_array, loc_left.unsqueeze(0)], dim=0)
                loc_right_array = torch.cat([loc_right_array, loc_right.unsqueeze(0)], dim=0)
                heading_array = torch.cat([heading_array, heading.unsqueeze(0)], dim=0)

                latlon_array = torch.cat([latlon_array, latlon.unsqueeze(0)], dim=0)
                # location (absolute) of the GPS device. It is given for evaluation.

                if self.stereo:
                    grd_right_imgs = torch.cat([grd_right_imgs, grd_img_right.unsqueeze(0)], dim=0)

        loc_shift_left = loc_left_array - loc_left_array[0:1, :]  # [N, 2], relative to the query location
        loc_shift_right = loc_right_array - loc_left_array[0:1, :]  # [N, 2], relative to the query location

        return left_camera_k, right_camera_k, grd_left_imgs, grd_right_imgs, \
               loc_shift_left, loc_shift_right, heading_array, loc_left_array, file_name


class SatDataset1(Dataset):  # without distractor, each satellite image corresponds to at least one grd image
    def __init__(self, root, gap_map=False,
                 transform=None, use_polar_sat=0):
        self.root = root
        # self.stereo = stereo
        # self.sequence = sequence

        if transform != None:
            self.satmap_transform = transform
            # self.grdimage_transform = transform[1]

        if use_polar_sat:
            self.test_sat_dir = 'satmap_polar/test'
        else:
            self.test_sat_dir = 'satmap/test'

        with open(test_file, 'r') as f:
            file_name = f.readlines()

        file_name = [file[:-1] for file in file_name]
        self.file_name = self.process_data(file_name)

        return

    def process_data(self, file_name):

        files = []
        for file in file_name:
            name, lat, lon = file.strip().split(' ')
            day, drive_dir, image_name = name.split('/')
            satFile = os.path.join(self.root, self.test_sat_dir, drive_dir, lat + '_' + lon + '.png')
            if satFile not in files:
                files.append(satFile)

        # latlon_to_satFile_dict = {}
        # for file in file_name:
        #     satFile, lat, lon = file.strip().split(' ')
        #     key = (lat, lon)
        #     if key not in latlon_to_satFile_dict.keys():
        #         latlon_to_satFile_dict[key] = satFile
        #
        # files = []
        # for key, val in latlon_to_satFile_dict.items():
        #     lat, lon = key
        #     satFile = val
        #     files.append((satFile, lat, lon))

        return files

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        # file_name, sat_lat, sat_lon = self.file_name[idx]
        SatMap_name = self.file_name[idx]
        sat_lat, sat_lon = os.path.basename(SatMap_name).split('.png')[0].split('_')

        sat_x, sat_y = utils.gps2utm(float(sat_lat), float(sat_lon))
        loc_sat = torch.tensor(np.array([sat_x, sat_y]))

        # =================== read satellite map ===================================

        # SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # x, y = utils.get_camera_gps_shift_left(heading_array[0].item())  # shift <1.4m
        # meter_per_pixel = utils.get_meter_per_pixel(scale=1)
        # shift_xy = (np.array([x, y]) / meter_per_pixel).astype(np.int32)
        if 'polar' in self.test_sat_dir:
            # crop out the central region
            SatMap_sidelength = utils.get_original_satmap_sidelength()
            width, height = sat_map.size
            centroid_x = width // 2 #+ shift_xy[0]
            centroid_y = height // 2 #- shift_xy[1]
            left = centroid_x - SatMap_sidelength / 2
            top = centroid_y - SatMap_sidelength / 2

            # crop
            sat_map = sat_map.crop((left, top, left + SatMap_sidelength, top + SatMap_sidelength))
        else:
            sat_map = sat_map.resize((1024, 256))

        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        return sat_map, loc_sat


class SatDataset2(Dataset):  # with distractor, there are many satellite images that do not have corresponding grd images.
    def __init__(self, root, transform=None, use_polar_sat=0):
        self.root = root
        self.transform = transform

        if use_polar_sat:
            self.test_sat_dir = 'satmap_polar/test'
        else:
            self.test_sat_dir = 'satmap/test'

        self.file_name = []
        test_df = pd.read_csv(os.path.join(root, test_csv_file_name))
        ignore_df = pd.read_csv(os.path.join(root, ignore_csv_file_name))

        # get image & location information
        dirs = os.listdir(os.path.join(root, self.test_sat_dir))
        for subdir in dirs:
            # subdir: such as 2011_09_26_drive_0019_sync
            if 'drive' not in subdir:
                continue

            if subdir in ignore_df.values:
                continue

            if subdir not in test_df.values:
                continue

            items = os.listdir(os.path.join(root, self.test_sat_dir, subdir))
            # order items
            items.sort()
            for item in items:
                if 'png' not in item.lower():
                    continue

                SatMap_name = os.path.join(subdir, item)
                self.file_name.append(SatMap_name)

        return

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        # day_dir is first 10 chat of file name
        file_name = self.file_name[idx]

        # get location
        gps = file_name.strip().split('/')
        gps = gps[1].strip().split('_')
        # latlon = [float(gps[0]), float(gps[1].strip('.png'))]
        # latlon = torch.from_numpy(np.asarray(latlon))
        utm_x, utm_y = utils.gps2utm(float(gps[0]), float(gps[1].strip('.png')))
        location = torch.from_numpy(np.asarray([utm_x, utm_y]))

        # get satmap image
        SatMap_name = os.path.join(self.root, self.test_sat_dir, file_name)
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

            if "polar" not in self.test_sat_dir:
                # crop satmap in center
                SatMap_sidelength = utils.get_original_satmap_sidelength()
                width, height = sat_map.size
                centroid_x = width // 2
                centroid_y = height // 2
                left = centroid_x - SatMap_sidelength / 2
                top = centroid_y - SatMap_sidelength / 2

                # crop
                sat_map = sat_map.crop((left, top, left + SatMap_sidelength, top + SatMap_sidelength))
            else:
                sat_map = sat_map.resize((1024, 256))

            # transform
            if self.transform is not None:
                sat_map = self.transform(sat_map)

        return sat_map, location


class DistanceBatchSampler:
    def __init__(self, sampler, batch_size, drop_last, required_dis, file_name):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.required_dis = required_dis
        self.backup = []
        self.backup_location = torch.tensor([])
        self.file_name = file_name

    def check_add(self, cur_location, location_list):
        if location_list.size()[0] > 0:
            dis = utils.gps2distance(cur_location[0], cur_location[1], location_list[:, 0], location_list[:, 1])
            if torch.min(dis) < self.required_dis:
                return False
        return True

    def __iter__(self):
        batch = []
        location_list = torch.tensor([])

        for idx in self.sampler:
            # check the idx gps location, not less than required distance

            # get location
            file_name = self.file_name[idx].strip().split(' ')[0]
            drive_dir = file_name[:38]
            image_no = file_name[38:]
            # oxt: such as 0000000000.txt
            oxts_file_name = os.path.join(root_dir, grdimage_dir, drive_dir, oxts_dir,
                                          image_no.lower().replace('.png', '.txt'))
            with open(oxts_file_name, 'r') as f:
                content = f.readline().split(' ')

                # get location
                cur_location = [float(content[0]), float(content[1])]
                cur_location = torch.from_numpy(np.asarray(cur_location))

                if self.check_add(cur_location, location_list):
                    # add to batch
                    batch.append(idx)
                    location_list = torch.cat([location_list, cur_location.unsqueeze(0)], dim=0)
                else:
                    # add to back up
                    self.backup.append(idx)
                    self.backup_location = torch.cat([self.backup_location, cur_location.unsqueeze(0)], dim=0)

            if len(batch) == self.batch_size:
                yield batch
                batch = []
                location_list = torch.tensor([])

                # pop back up
                remove = []
                for i in range(len(self.backup)):
                    idx = self.backup[i]
                    cur_location = self.backup_location[i]

                    if self.check_add(cur_location, location_list):
                        # add to batch
                        batch.append(idx)
                        location_list = torch.cat([location_list, cur_location.unsqueeze(0)], dim=0)

                        # need remove from backup
                        remove.append(i)

                for i in sorted(remove, reverse=True):
                    if i == len(self.backup) - 1:
                        # last item
                        self.backup_location = self.backup_location[:i]
                    else:
                        self.backup_location = torch.cat((self.backup_location[:i], self.backup_location[i + 1:]))
                    self.backup.remove(self.backup[i])
                # print('left in backup:',len(self.backup),self.backup_location.size())

        if len(batch) > 0 and not self.drop_last:
            yield batch
            print('batched all, left in backup:', len(self.backup), self.backup_location.size())

    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore


