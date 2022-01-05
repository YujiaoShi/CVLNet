from torch.utils.data import DataLoader
from torchvision import transforms
import time
from dataLoader.datasets import *

visualise_debug = True


def load_data(file, batch_size, stereo, sequence, shift_range=0,
                    use_polar_sat=0, use_project_grd=0, use_semantic=0):
    SatMap_process_sidelength = utils.get_process_satmap_sidelength()

    if use_polar_sat:
        satmap_transform = transforms.Compose([
            transforms.Resize(size=[128, 512]),
            transforms.ToTensor(),
        ])
    else:
        satmap_transform = transforms.Compose([
            transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
            transforms.ToTensor(),
        ])

    if use_project_grd:
        Grd_h = Grd_w = SatMap_process_sidelength
    else:
        Grd_h = GrdImg_H
        Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])
    if use_polar_sat:
        grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h//2, Grd_w//2]),
        transforms.ToTensor(),
    ])

    train_set = SatGrdDataset(root=root_dir, file_name=file, stereo=stereo, sequence=sequence,
                              transform=(satmap_transform, grdimage_transform),
                              use_polar_sat=use_polar_sat, use_project_grd=use_project_grd, use_semantic=use_semantic)
    # if shift_range > 0:
    meter_per_pixel = utils.get_meter_per_pixel()
    shift_meter = (shift_range * 2 + 1) * meter_per_pixel * 512 / 32
    file_name = train_set.get_file_list()
    bs = DistanceBatchSampler(torch.utils.data.RandomSampler(train_set), batch_size, True, shift_meter, file_name)
    train_loader = DataLoader(train_set, batch_sampler=bs, num_workers=num_thread_workers)
    # else:
    #     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
    #                               num_workers=num_thread_workers, drop_last=False)
    return train_loader


def load_train_data(batch_size, stereo, sequence, shift_range=0,
                    use_polar_sat=0, use_project_grd=0):
    SatMap_process_sidelength = utils.get_process_satmap_sidelength()
    if use_polar_sat:
        satmap_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # 0~1 to -1~1
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif visualise_debug:
        satmap_transform = transforms.Compose([
            transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
            transforms.ToTensor(),
        ])
    else:
        satmap_transform = transforms.Compose([
            transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # 0~1 to -1~1
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if use_project_grd:
        Grd_h = Grd_w = SatMap_process_sidelength
    else:
        Grd_h = GrdImg_H
        Grd_w = GrdImg_W


    if visualise_debug:
        grdimage_transform = transforms.Compose([
            transforms.Resize(size=[Grd_h, Grd_w]),
            transforms.ToTensor(),
        ])
    else:
        grdimage_transform = transforms.Compose([
            transforms.Resize(size=[Grd_h, Grd_w]),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # 0~1 to -1~1
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    train_set = SatGrdDataset(root=root_dir, stereo=stereo, sequence=sequence,
                              transform=(satmap_transform, grdimage_transform),
                              use_polar_sat=use_polar_sat, use_project_grd=use_project_grd)
    if shift_range > 0:
        meter_per_pixel = utils.get_meter_per_pixel()
        shift_meter = (shift_range * 2 + 1) * meter_per_pixel * 512 / 32
        file_name = train_set.get_file_list()
        bs = DistanceBatchSampler(torch.utils.data.RandomSampler(train_set), batch_size, True, shift_meter, file_name)
        train_loader = DataLoader(train_set, batch_sampler=bs, num_workers=num_thread_workers)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  num_workers=num_thread_workers, drop_last=False)
    return train_loader


def load_test_sat_data2(batch_size, use_polar_sat=0):
    if use_polar_sat:
        satmap_transform = transforms.Compose([
            transforms.Resize(size=[128, 512]),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # 0~1 to -1~1
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        SatMap_process_sidelength = utils.get_process_satmap_sidelength()
        satmap_transform = transforms.Compose([
            transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # 0~1 to -1~1
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    test_set = SatDataset2(root=root_dir, transform=satmap_transform, use_polar_sat=use_polar_sat)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                             num_workers=num_thread_workers, drop_last=False)

    return test_loader

def load_test_sat_data1(batch_size, gap_map=True, use_polar_sat=0):
    SatMap_process_sidelength = utils.get_process_satmap_sidelength()
    if use_polar_sat:
        satmap_transform = transforms.Compose([
            transforms.Resize(size=[128, 512]),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # 0~1
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif visualise_debug:
        satmap_transform = transforms.Compose([
            transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
            transforms.ToTensor(),
        ])
    else:
        satmap_transform = transforms.Compose([
            transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # 0~1 to -1~1
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    test_dataset = SatDataset1(root=root_dir, gap_map=gap_map, transform=satmap_transform,
                               use_polar_sat=use_polar_sat)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                             num_workers=num_thread_workers, drop_last=False)
    return test_loader

def load_test_grd_data(batch_size, stereo, sequence, use_project_grd=0, use_polar_sat=0,
                                use_semantic=0):
    SatMap_process_sidelength = utils.get_process_satmap_sidelength()

    if use_project_grd:
        Grd_h = Grd_w = SatMap_process_sidelength
    else:
        Grd_h = GrdImg_H
        Grd_w = GrdImg_W

    if visualise_debug:
        grdimage_transform = transforms.Compose([
            transforms.Resize(size=[Grd_h, Grd_w]),
            transforms.ToTensor(),
        ])
    else:
        grdimage_transform = transforms.Compose([
            transforms.Resize(size=[Grd_h, Grd_w]),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # 0~1 to -1~1
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if use_polar_sat:
        grdimage_transform = transforms.Compose([
            transforms.Resize(size=[Grd_h//2, Grd_w//2]),
            transforms.ToTensor(),
        ])

    test_dataset = GrdDataset(root=root_dir, stereo=stereo, sequence=sequence,
                                 transform=grdimage_transform,
                                 use_project_grd=use_project_grd, use_semantic=use_semantic)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                             num_workers=num_thread_workers, drop_last=False)
    return test_loader



if __name__ == '__main__':
    # test to load 1 data
    # train_loader = load_train_data(8,stereo=True, sequence=8)
    # for i, data in enumerate(train_loader, 0):
    #     clock = time.process_time()
    #     # left_camera_k,sat_map, grd_img_left, coarse_loc, heading, right_camera_k, grd_img_right = data
    #     left_camera_k, sat_map, grd_img_left, loc_left, heading, right_camera_k, grd_img_right, loc_right, latlon_array\
    #         = data
    #     print('load clock:{:.2f}'.format(time.process_time()-clock))
    #     print(heading.size(), loc_left.size())

    _, sat2_set = load_test_sat_data2(1)
    # _, grd_set = load_test_grd_data(1, 0, 1, use_project_grd=0, use_polar_sat=0)
    #
    sat_files = sat2_set.file_name
    with open('database_sat2_files.txt', 'a') as f:
        for file in sat_files:
            f.write(file + '\n')
    
    # grd_files = grd_set.file_name
    #
    # a = 1

    # test_cnt = 0
    # for i, data in enumerate(test_loader, 0):
    #     left_camera_k, sat_map, grd_img_left, loc_left, heading, right_camera_k, grd_img_right, loc_right = data
    #     # print(heading.size(), loc_left.size())
    #     test_cnt += i
    # print("load all, cnt:", test_cnt)

