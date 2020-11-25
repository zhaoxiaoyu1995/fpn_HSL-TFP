import scipy.io as scio
import numpy as np
import os
from torch.utils.data import Dataset
import torch
import cv2


class MyDataset(Dataset):
    def __init__(self, root, transform_func, resize_shape):
        self.root = root
        self.mats = sorted(os.listdir(self.root))
        self.transform_func = transform_func
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.mats)

    def __getitem__(self, item):
        mat = self.mats[item]
        layout_map = np.zeros((200, 200))
        datadict = scio.loadmat(self.root+mat)
        location = datadict['list'][0]
        for i in location:
            i = i - 1
            layout_map[(i % 10) * 20:(i % 10) * 20 + 20, i // 10 * 20:i // 10 * 20 + 20] = np.ones((20, 20))
        heat_map = datadict['u']
        return self.transform_func(layout_map, heat_map, self.resize_shape)


class MyDataset_diff(Dataset):
    def __init__(self, root, transform_func, resize_shape):
        self.root = root
        self.mats = sorted(os.listdir(self.root))
        self.transform_func = transform_func
        self.resize_shape = resize_shape

        data_dict_normal = scio.loadmat("data/problem2/Example_normal.mat")
        self.heat_map_normal = data_dict_normal['u']
        location = data_dict_normal['list'][0]
        layout_map = np.zeros((200, 200))
        for i in location:
            i = i - 1
            layout_map[(i % 10) * 20:(i % 10) * 20 + 20, i // 10 * 20:i // 10 * 20 + 20] = np.ones((20, 20))
        self.layout_map_normal = layout_map

    def __len__(self):
        return len(self.mats)

    def __getitem__(self, item):
        mat = self.mats[item]
        layout_map = np.zeros((200, 200))
        datadict = scio.loadmat(self.root + mat)
        location = datadict['list'][0]
        for i in location:
            i = i - 1
            layout_map[(i % 10) * 20:(i % 10) * 20 + 20, i // 10 * 20:i // 10 * 20 + 20] = np.ones((20, 20))
        heat_map = datadict['u']
        return self.transform_func(layout_map - self.layout_map_normal, heat_map - self.heat_map_normal, self.resize_shape)


# 原始版本：温度值规范处理：(u-260) / 100
class MyDataset_basic(Dataset):
    def __init__(self, root, transform_func, resize_shape):
        self.root = root
        self.mats = sorted(os.listdir(self.root))
        self.transform_func = transform_func
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.mats)

    def __getitem__(self, item):
        mat = self.mats[item]
        layout_map = np.zeros((200, 200))
        datadict = scio.loadmat(self.root+mat)
        location = datadict['list'][0]
        for i in location:
            i = i - 1
            layout_map[(i % 10) * 20:(i % 10) * 20 + 20, i // 10 * 20:i // 10 * 20 + 20] = np.ones((20, 20))
        heat_map = (datadict['u'] - 260) / 100
        return self.transform_func(layout_map, heat_map, self.resize_shape)


class GeneralDataset(MyDataset):
    def __init__(self, transform_func, resize_shape):
        super().__init__('/home/zhaoxiaoyu/data/transfer_data/problem1/train/train500/', transform_func, resize_shape)


class ValDataset(MyDataset):
    def __init__(self, transform_func, resize_shape):
        super().__init__('/home/zhaoxiaoyu/data/transfer_data/problem1/val/val500/', transform_func, resize_shape)


class TestDataset(MyDataset):
    def __init__(self, transform_func, resize_shape):
        super().__init__('/home/zhaoxiaoyu/data/transfer_data/problem1/test/', transform_func, resize_shape)


class GeneralDataset_diff(MyDataset_diff):
    def __init__(self, transform_func, resize_shape):
        super().__init__('/home/zhaoxiaoyu/data/transfer_data/problem1/train/train500/', transform_func, resize_shape)


class ValDataset_diff(MyDataset_diff):
    def __init__(self, transform_func, resize_shape):
        super().__init__('/home/zhaoxiaoyu/data/transfer_data/problem2/val/val50/', transform_func, resize_shape)


class TestDataset_diff(MyDataset_diff):
    def __init__(self, transform_func, resize_shape):
        super().__init__('/home/zhaoxiaoyu/data/transfer_data/problem1/test/', transform_func, resize_shape)


class GeneralDataset_basic(MyDataset_basic):
    def __init__(self, transform_func, resize_shape):
        super().__init__('/home/zhaoxiaoyu/data/source_data/problem1/train/', transform_func, resize_shape)


class ValDataset_basic(MyDataset_basic):
    def __init__(self, transform_func, resize_shape):
        super().__init__('/home/zhaoxiaoyu/data/source_data/problem1/val/', transform_func, resize_shape)


class TestDataset_basic(MyDataset_basic):
    def __init__(self, transform_func, resize_shape):
        super().__init__('/home/zhaoxiaoyu/data/source_data/problem1/test/', transform_func, resize_shape)


def trans_stack(layout_map, heat, resize_shape):
    res = np.vstack((layout_map, heat))
    res = cv2.resize(res, resize_shape)
    res= np.expand_dims(res, 0)
    return torch.from_numpy(res.astype(np.float32))


def trans_concat(layout_map, heat_map, resize_shape):
    res = np.array([cv2.resize(layout_map, resize_shape), cv2.resize(heat_map, resize_shape)])
    return torch.from_numpy(res.astype(np.float32))


def trans_separate(layout_map, heat_map, resize_shape):
    layout_map = np.expand_dims(cv2.resize(layout_map, resize_shape), 0)
    heat_map = np.expand_dims(cv2.resize(heat_map, resize_shape), 0)
    return torch.from_numpy(layout_map.astype(np.float32)), torch.from_numpy(heat_map.astype(np.float32))


if __name__ == "__main__":
    dataset = GeneralDataset(trans_separate, (200, 200))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    for layout_map, heat_map in data_loader:
        print(heat_map.shape)
        print(torch.max(heat_map), torch.min(heat_map))
        break
