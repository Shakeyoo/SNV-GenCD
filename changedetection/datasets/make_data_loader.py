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
class SingleChangeDetectionDatset(Dataset):
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

    def __transforms(self, aug, pre_img, post_img, cd_label, t1_label):
        if aug:
            pre_img, post_img, cd_label, t1_label = imutils.random_crop_mcd_1(pre_img, post_img, cd_label, t1_label,self.crop_size)
            pre_img, post_img, cd_label, t1_label = imutils.random_fliplr_mcd(pre_img, post_img, cd_label, t1_label)
            pre_img, post_img, cd_label, t1_label = imutils.random_flipud_mcd(pre_img, post_img, cd_label, t1_label)
            pre_img, post_img, cd_label, t1_label = imutils.random_rot_mcd(pre_img, post_img, cd_label, t1_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, cd_label, t1_label

    def __getitem__(self, index):
        if 'train' in self.data_pro_type:
            pre_path = os.path.join(self.dataset_path, 'A', self.data_list[index] )
            post_path = os.path.join(self.dataset_path, 'B', self.data_list[index])
            T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index] )
            #T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index])
            cd_label_path = os.path.join(self.dataset_path, 'label', self.data_list[index])
        else:
            pre_path = os.path.join(self.dataset_path, 'A', self.data_list[index])
            post_path = os.path.join(self.dataset_path, 'B', self.data_list[index])
            T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index])
            #T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index])
            cd_label_path = os.path.join(self.dataset_path, 'label', self.data_list[index])

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        t1_label = self.loader(T1_label_path)
        t1_label = t1_label/255
        #t2_label = self.loader(T2_label_path)
        cd_label = self.loader(cd_label_path)
        cd_label = cd_label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, cd_label, t1_label = self.__transforms(True, pre_img, post_img, cd_label, t1_label)
        else:
            pre_img, post_img, cd_label, t1_label= self.__transforms(False, pre_img, post_img, cd_label, t1_label)
            cd_label = np.asarray(cd_label)
            t1_label = np.asarray(t1_label)
            #t2_label = np.asarray(t2_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, cd_label, t1_label, data_idx

    def __len__(self):
        return len(self.data_list)
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

class SemanticChangeDetectionDatset2(Dataset):
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

    def __transforms(self, aug, pre_img, post_img, cd_label,  t2_label):
        if aug:
            pre_img, post_img, cd_label,  t2_label = imutils.random_crop_mcd(pre_img, post_img, cd_label,  t2_label, self.crop_size)
            pre_img, post_img, cd_label,  t2_label = imutils.random_fliplr_mcd(pre_img, post_img, cd_label,  t2_label)
            pre_img, post_img, cd_label,  t2_label = imutils.random_flipud_mcd(pre_img, post_img, cd_label,  t2_label)
            pre_img, post_img, cd_label,  t2_label = imutils.random_rot_mcd(pre_img, post_img, cd_label,  t2_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, cd_label,  t2_label

    def __getitem__(self, index):
        if 'train' in self.data_pro_type:
            pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index] + '.png')
            post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index] + '.png')
            T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index] + '.png')
            cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index] + '.png')
        else:
            pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
            post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
            T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index])
            cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index])

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        t2_label = self.loader(T2_label_path)
        cd_label = self.loader(cd_label_path)
        cd_label = cd_label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, cd_label, t2_label = self.__transforms(True, pre_img, post_img, cd_label, t1_label, t2_label)
        else:
            pre_img, post_img, cd_label, t2_label = self.__transforms(False, pre_img, post_img, cd_label, t1_label, t2_label)
            cd_label = np.asarray(cd_label)
            t2_label = np.asarray(t2_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, cd_label,t2_label, data_idx

    def __len__(self):
        return len(self.data_list)
class ConChangeDetectionDatset(Dataset):
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
        # 用 repr() 打印可见隐藏字符
        #print("Pre image path:", repr(pre_path))
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

class ChangeDetectionDatset2(Dataset):
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

    def __transforms(self, aug, pre_img, mid_img,post_img, label):
        if aug:
            pre_img, mid_img,post_img, label = imutils.random_crop_new2(pre_img, mid_img,post_img, label, self.crop_size)
            pre_img, mid_img,post_img, label = imutils.random_fliplr2(pre_img, mid_img,post_img, label)
            pre_img, mid_img,post_img, label = imutils.random_flipud2(pre_img, mid_img,post_img, label)
            pre_img, mid_img,post_img, label = imutils.random_rot2(pre_img, mid_img,post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        mid_img = imutils.normalize_img(mid_img)  # imagenet normalization
        mid_img = np.transpose(mid_img, (2, 0, 1))

        return pre_img, mid_img, post_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'A', self.data_list[index])
        mid_path = os.path.join(self.dataset_path, 'M', self.data_list[index])
        post_path = os.path.join(self.dataset_path, 'B', self.data_list[index])
        label_path = os.path.join(self.dataset_path, 'label', self.data_list[index])
        # 用 repr() 打印可见隐藏字符
        #print("Pre image path:", repr(pre_path))
        pre_img = self.loader(pre_path)
        mid_img= self.loader(mid_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path)
        label = label / 255

        if 'train' in self.data_pro_type:
            pre_img, mid_img, post_img, label = self.__transforms(True, pre_img, mid_img,post_img, label)
        else:
            pre_img, mid_img,post_img, label = self.__transforms(False, pre_img, mid_img,post_img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, mid_img,post_img, label, data_idx
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



class SemanticChangeDetectionDatset(Dataset):
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

    def __transforms(self, aug, pre_img, post_img, cd_label, t1_label, t2_label):
        if aug:
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_crop_mcd(pre_img, post_img, cd_label, t1_label, t2_label, self.crop_size)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_fliplr_mcd(pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_flipud_mcd(pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_rot_mcd(pre_img, post_img, cd_label, t1_label, t2_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, cd_label, t1_label, t2_label

    def __getitem__(self, index):
        if 'train' in self.data_pro_type:
            pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index] + '.png')
            post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index] + '.png')
            T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index] + '.png')
            T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index] + '.png')
            cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index] + '.png')
        else:
            pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
            post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
            T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index])
            T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index])
            cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index])

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        t1_label = self.loader(T1_label_path)
        t2_label = self.loader(T2_label_path)
        cd_label = self.loader(cd_label_path)
        cd_label = cd_label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, cd_label, t1_label, t2_label = self.__transforms(True, pre_img, post_img, cd_label, t1_label, t2_label)
        else:
            pre_img, post_img, cd_label, t1_label, t2_label = self.__transforms(False, pre_img, post_img, cd_label, t1_label, t2_label)
            cd_label = np.asarray(cd_label)
            t1_label = np.asarray(t1_label)
            t2_label = np.asarray(t2_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, cd_label, t1_label, t2_label, data_idx

    def __len__(self):
        return len(self.data_list)


class DamageAssessmentDatset(Dataset):
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

    def __transforms(self, aug, pre_img, post_img, loc_label, clf_label):
        if aug:
            pre_img, post_img, loc_label, clf_label = imutils.random_crop_bda(pre_img, post_img, loc_label, clf_label, self.crop_size)
            pre_img, post_img, loc_label, clf_label = imutils.random_fliplr_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_flipud_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_rot_bda(pre_img, post_img, loc_label, clf_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, loc_label, clf_label

    def __getitem__(self, index):
        if 'train' in self.data_pro_type: 
            parts = self.data_list[index].rsplit('_', 2)

            pre_img_name = f"{parts[0]}_pre_disaster_{parts[1]}_{parts[2]}.png"
            post_img_name = f"{parts[0]}_post_disaster_{parts[1]}_{parts[2]}.png"

            pre_path = os.path.join(self.dataset_path, 'images', pre_img_name)
            post_path = os.path.join(self.dataset_path, 'images', post_img_name)
            
            loc_label_path = os.path.join(self.dataset_path, 'masks', pre_img_name)
            clf_label_path = os.path.join(self.dataset_path, 'masks', post_img_name)
        else:
            pre_path = os.path.join(self.dataset_path, 'images', self.data_list[index] + '_pre_disaster.png')
            post_path = os.path.join(self.dataset_path, 'images', self.data_list[index] + '_post_disaster.png')
            loc_label_path = os.path.join(self.dataset_path, 'masks', self.data_list[index]+ '_pre_disaster.png')
            clf_label_path = os.path.join(self.dataset_path, 'masks', self.data_list[index]+ '_post_disaster.png')

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        loc_label = self.loader(loc_label_path)[:,:,0]
        clf_label = self.loader(clf_label_path)[:,:,0]

        if 'train' in self.data_pro_type:
            pre_img, post_img, loc_label, clf_label = self.__transforms(True, pre_img, post_img, loc_label, clf_label)
            clf_label[clf_label == 0] = 255
        else:
            pre_img, post_img, loc_label, clf_label = self.__transforms(False, pre_img, post_img, loc_label, clf_label)
            loc_label = np.asarray(loc_label)
            clf_label = np.asarray(clf_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)


class BuildingDatset(Dataset):
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

    def __transforms(self, aug, img, loc_label):
        if aug:
            img, loc_label = imutils.random_crop_build(img,loc_label,self.crop_size)
            img, loc_label = imutils.random_fliplr_build(img, loc_label)
            img, loc_label = imutils.random_flipud_build(img, loc_label)
            img, loc_label = imutils.random_rot_build(img, loc_label)

        img = imutils.normalize_img(img)  # imagenet normalization
        img = np.transpose(img, (2, 0, 1))

        return img, loc_label

    def __getitem__(self, index):
        # 根据数据处理类型构造图像和标签路径
        if 'train' in self.data_pro_type:
            img_name = self.data_list[index]
            img_path = os.path.join(self.dataset_path, 'images', img_name)
            label_path = os.path.join(self.dataset_path, 'mask', img_name)
        else:
            img_name = self.data_list[index]
            img_path = os.path.join(self.dataset_path, 'images', img_name)
            label_path = os.path.join(self.dataset_path, 'mask', img_name)

        # 加载图像和标签
        img = self.loader(img_path)
        label = self.loader(label_path)

        # 数据增强和处理
        if 'train' in self.data_pro_type:
            img, label = self.__transforms(True, img, label)
        else:
            img, label = self.__transforms(False, img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return img, label, data_idx

    def __len__(self):
        return len(self.data_list)


def make_data_loader(args, **kwargs):  # **kwargs could be omitted
    if 'xview2' in args.dataset :
        dataset = PseudoTemporalDataset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader
    if 'SYSU' in args.dataset or 'LEVIR-CD+' in args.dataset or 'WHU' in args.dataset:
        dataset = ChangeDetectionDatset2(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader
    elif 'xBD' in args.dataset:
        dataset = BuildingDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader
    
    elif 'SECOND' in args.dataset:
        dataset = SingleChangeDetectionDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader
    elif 'cd' in args.dataset:
        dataset = SingleChangeDetectionDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader
    
    else:
        raise NotImplementedError


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SECOND DataLoader Test")
    parser.add_argument('--dataset', type=str, default='WHUBCD')
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='D:/Workspace/Python/STCD/data/ST-WHU-BCD')
    parser.add_argument('--data_list_path', type=str, default='./ST-WHU-BCD/train_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--data_name_list', type=list)

    args = parser.parse_args()

    with open(args.data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.data_name_list = data_name_list
    train_data_loader = make_data_loader(args)
    for i, data in enumerate(train_data_loader):
        pre_img, post_img, labels, _ = data
        pre_data, post_data = Variable(pre_img), Variable(post_img)
        labels = Variable(labels)
        print(i, "个inputs", pre_data.data.size(), "labels", labels.data.size())

def generate_target(imgs, labels):
    imgs_nums = imgs.size(0)  # 一组里面有多少张图像
    org_inds = torch.arange(imgs_nums)
    t = 0
    while True and t <= 50:  # 构造伪配对序列
        t += 1
        shuffle_inds = torch.randperm(imgs_nums)

        ok = org_inds == shuffle_inds
        if torch.all(~ok):
            break
    virtual_imgs2 = imgs[shuffle_inds]
    virtual_labels2 = labels[shuffle_inds]
    return imgs, labels, virtual_imgs2, virtual_labels2

