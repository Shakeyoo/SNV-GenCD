import argparse
import os

import imageio
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

import changedetection.datasets.imutils as imutils


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img


def one_hot_encoding(image, num_classes=8):
    # Create a one hot encoded tensor
    one_hot = np.eye(num_classes)[image.astype(np.uint8)]

    # Move the channel axis to the front
    # one_hot = np.moveaxis(one_hot, -1, 0)

    return one_hot

def numpyoneTo255(label_img):
    image_arr = np.array(label_img)
    image_arr[image_arr == 1] = 255
    label_img = Image.fromarray(image_arr)
    return label_img
    
class PseudoTemporalDataset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, label):
        if aug:
            pre_img, label = imutils.single_random_crop_new(pre_img,  label, self.crop_size)
            pre_img, label = imutils.single_random_fliplr(pre_img, label)
            pre_img, label = imutils.single_random_flipud(pre_img, label)
            pre_img,  label = imutils.single_random_rot(pre_img,  label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))



        return pre_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
        #post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
        label_path = os.path.join(self.dataset_path, 'T1_GT', self.data_list[index])
        pre_img = self.loader(pre_path)
        #post_img = self.loader(post_path)
        label = self.loader(label_path)
        #label = label / 255
        #label=label*255

        if 'train' in self.data_pro_type:
            pre_img, label = self.__transforms(True, pre_img,  label)
        else:
            pre_img,  label = self.__transforms(False, pre_img,  label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        #label=numpyoneTo255(label)
        return pre_img,  label, data_idx

    def __len__(self):
        return len(self.data_list)


class ConChangeDetectionDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.data_list = [x.strip().strip('`') for x in data_list if x.strip()]

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, lama_img,label):
        if aug:
            pre_img, post_img, lama_img,label = imutils.con_random_crop_new(pre_img, post_img, lama_img,label, self.crop_size)
            pre_img, post_img, lama_img,label = imutils.con_random_fliplr(pre_img, post_img, lama_img,label)
            pre_img, post_img, lama_img,label = imutils.con_random_flipud(pre_img, post_img, lama_img,label)
            pre_img, post_img, lama_img,label = imutils.con_random_rot(pre_img, post_img,lama_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        lama_img = imutils.normalize_img(lama_img)
        lama_img = np.transpose(lama_img, (2, 0, 1))

        return pre_img, post_img, lama_img,label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
        post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
        lama_path = os.path.join(self.dataset_path, 'lama', self.data_list[index])
        label_path = os.path.join(self.dataset_path, 'GT', self.data_list[index])
        
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        lama_img= self.loader(lama_path)
        label = self.loader(label_path)
        label = label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, lama_img,label = self.__transforms(True, pre_img, post_img, lama_img,label)
        else:
            pre_img, post_img, lama_img,label = self.__transforms(False, pre_img, post_img, lama_img,label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, post_img,lama_img,label, data_idx

    def __len__(self): 
        return len(self.data_list)

class ChangeDetectionDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        # 清洗文件名中的异常字符
        self.data_list = [x.strip().strip('`') for x in data_list if x.strip()]

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'A', self.data_list[index])
        post_path = os.path.join(self.dataset_path, 'B', self.data_list[index])
        label_path = os.path.join(self.dataset_path, 'label', self.data_list[index])
        # 用 repr() 打印可见隐藏字符
        #print("Pre image path:", repr(pre_path))
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path)
        label = label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, label = self.__transforms(True, pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.__transforms(False, pre_img, post_img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, post_img, label, data_idx

    def __len__(self):
        return len(self.data_list)


def make_data_loader(args, **kwargs):  # **kwargs could be omitted
    if 'LEVIR-CD' in args.dataset :
        dataset = PseudoTemporalDataset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader
    else:
        raise NotImplementedError



