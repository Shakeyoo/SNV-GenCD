import sys

import wandb

sys.path.append('/media/mcislab/WYY/project/MambaCD')

import argparse
import os
import time

import numpy as np
from changedetection.data.transform import crop, hflip, normalize, resize, blur, cutout
from torch.utils.data import Dataset
from torchvision import transforms
from changedetection.configs.config import get_config
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from changedetection.datasets.make_data_loader import ChangeDetectionDatset, make_data_loader
from changedetection.utils_func.metrics import Evaluator
from changedetection.models.MambaBCD import STMambaBCD
import random
import cv2
from PIL import Image


import changedetection.utils_func.lovasz_loss as L
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def comtrastive_loss(pred, target, mean=False):
    output_pred = F.softmax(pred, dim=1)
    postive_pred = output_pred[:2]
    negtive_pred = output_pred[2:]
    M = target[:2].clone().float()

    loss_pos = F.mse_loss(postive_pred[:, 0, :, :], negtive_pred[:, 0, :, :], reduction='none') * (1 - M)
    loss_neg1 = F.mse_loss(postive_pred[:, 0, :, :], negtive_pred[:, 1, :, :], reduction='none') * M
    loss_neg2 = F.mse_loss(postive_pred[:, 1, :, :], negtive_pred[:, 0, :, :], reduction='none') * M

    loss_ct = loss_pos + loss_neg1 + loss_neg2
    if mean:
        loss_ct = loss_ct.mean()
    return loss_ct
class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)

        self.evaluator = Evaluator(num_class=2)

        self.deep_model = STMambaBCD(
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=config.MODEL.VSSM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VSSM.DEPTHS,
            dims=config.MODEL.VSSM.EMBED_DIM,
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            )
        self.deep_model = self.deep_model.cuda()

        #保存模型路径
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                             '实验/mylevircd' )

        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(self.MEAN, self.STD)

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
                 

    def training(self):
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)

        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
        # 从数据加载器中获取图像对与标签，假设 image_A 和 image_B 形状为 [B, 3, 256, 256]，B=2
            image_A, image_B, label, _ = data  
        # 保留原始输入
            pre_change_imgs, post_change_imgs, labels = image_A, image_B, label
        
        # 打印调试，确认形状
            #print("image_A shape:", image_A.shape)  # 如：[2, 3, 256, 256]
        
        # 对 batch 中每个样本独立进行数据增强，生成增强图像
            aug_images_A = []  # 用于存放 image_A 的增强版本
            aug_images_B = []  # 用于存放 image_B 的增强版本
            B = image_A.size(0)  # Batch size，此处 B=2
            for i in range(B):
            # 取当前样本 image_A[i]，形状 [3, 256, 256]
                single_image_A = image_A[i]  
            # 转置成 HWC 并转换为 numpy 数组
                single_image_A_np = single_image_A.permute(1, 2, 0).cpu().numpy()
            # 转换为 PIL Image
                pil_image_A = Image.fromarray(np.uint8(single_image_A_np))
            
            # 同理处理 image_B[i]
                single_image_B = image_B[i]
                single_image_B_np = single_image_B.permute(1, 2, 0).cpu().numpy()
                pil_image_B = Image.fromarray(np.uint8(single_image_B_np))
            
            # 对 PIL 图像应用一系列增强操作
            # 可根据随机概率选择增强策略
                if random.random() < 0.8:
                    pil_image_A = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(pil_image_A)
                    pil_image_B = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(pil_image_B)
                pil_image_A = transforms.RandomGrayscale(p=0.2)(pil_image_A)
                pil_image_B = transforms.RandomGrayscale(p=0.2)(pil_image_B)
                pil_image_A = blur(pil_image_A, p=0.5)  # 假设 blur 返回的是 PIL Image
                pil_image_B = blur(pil_image_B, p=0.5)
            # 进行 cutout 处理（cutout 同时对 image_A、image_B 与 label 进行处理）
                pil_image_A, pil_image_B, label_aug = cutout(pil_image_A, pil_image_B, label[i], p=0.5)
            
            # 将增强后的 PIL 图像转换回 tensor，并归一化
                aug_tensor_A = self.normalize(self.to_tensor(pil_image_A))  # 形状 [3, 256, 256]
                aug_tensor_B = self.normalize(self.to_tensor(pil_image_B))  # 形状 [3, 256, 256]
            
            # 加上 batch 维度后存入列表
                aug_images_A.append(aug_tensor_A.unsqueeze(0))
                aug_images_B.append(aug_tensor_B.unsqueeze(0))
        
        # 合并 batch 内所有样本，形成增强后的图像批次，形状为 [B, 3, 256, 256]
            aug_images_A = torch.cat(aug_images_A, dim=0)
            aug_images_B = torch.cat(aug_images_B, dim=0)
        # 对于增强样本，将其标签设为 0（无变化），以构造对比学习中的正样本对
            target_aug = labels * 0  # 形状 [B, 256, 256]
        
        # 构造新 batch：将原始图像对和增强图像对沿 batch 维度拼接
        # 原始 batch: pre_change_imgs, post_change_imgs, labels 形状均为 [B, 3, 256, 256] 或 [B, 256,256]
            A_l_aug = torch.cat((pre_change_imgs, aug_images_A), dim=0)   # 形状: [2B, 3, 256, 256] 例如 [4,3,256,256]
            B_l_aug = torch.cat((post_change_imgs, aug_images_B), dim=0)    # 同上
            target_l_aug = torch.cat((labels, target_aug), dim=0)           # 形状: [2B, 256, 256]
        
        # 移动到 GPU，并转换数据类型
            A_l_aug = A_l_aug.cuda().float()
            B_l_aug = B_l_aug.cuda()
            target_l_aug = target_l_aug.cuda().long()
        
        # Forward pass
            pred, _, _ = self.deep_model(A_l_aug, B_l_aug)
            pred2, _, _ = self.deep_model(B_l_aug, A_l_aug)
        
            self.optim.zero_grad()
        
        # 计算交叉熵损失
            ce_loss = F.cross_entropy(pred, target_l_aug, ignore_index=255)
            ce_loss2 = F.cross_entropy(pred2, target_l_aug, ignore_index=255)
            lovasz_loss_2 = L.lovasz_softmax(F.softmax(pred2, dim=1), target_l_aug, ignore=255)
            lovasz_loss = L.lovasz_softmax(F.softmax(pred, dim=1), target_l_aug, ignore=255)
            # 计算对比损失，假设 comtrastive_loss 已经定义好，且要求输入为模型的预测 logits 和目标
            contrast_loss_val = comtrastive_loss(pred, target_l_aug, mean=True)
            #contrast_loss_val2 = comtrastive_loss(pred2, target_l_aug, mean=True)
            
            final_loss = 0.5*(ce_loss + ce_loss2)+0.375*(lovasz_loss+lovasz_loss_2)+contrast_loss_val
            #final_loss=ce_loss2+0.75*lovasz_loss_2+contrast_loss_val
            #final_loss = ce_loss +0.75*lovasz_loss
            

            final_loss.backward()
            self.optim.step()

            # Log training loss to wandb
            # wandb.log({
            #     "Training Loss": final_loss.item(),
            #     "Cross Entropy Loss (Output 1)": ce_loss.item(),
            #     #"Cross Entropy Loss (Output 2)": ce_loss_2.item(),
            #     "Lovasz Loss (Output 1)": lovasz_loss.item(),
            #     "Lovasz Loss (Output 2)": lovasz_loss_2.item(),
            #     "Con Loss":contrast_loss_val.item(),
            #     "Final Loss":final_loss.item(),
            #     "Iteration": itera + 1
            # })

            if (itera + 1) % 10 == 0:
                print(f'iter is {itera + 1}, overall loss is {final_loss.item()}')

                if (itera + 1) % 300 == 0:
                    # Perform evaluation every 100 iterations
                    self.deep_model.eval()
                    rec, pre, oa, f1_score, iou, kc = self.validation()

                    # Log validation metrics to wandb
                    # wandb.log({
                    #     "Recall": rec,
                    #     "Precision": pre,
                    #     "Overall Accuracy": oa,
                    #     "F1 Score": f1_score,
                    #     "IoU": iou,
                    #     "Kappa Coefficient": kc
                    # })

                    if kc > best_kc:
                        # Save the model if performance improves
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'{itera + 1}_model.pth'))
                        # wandb.save(os.path.join(self.model_save_path, f'{itera + 1}_model.pth'))  # Save to wandb
                        best_kc = kc
                        best_round = [rec, pre, oa, f1_score, iou, kc]

                    self.deep_model.train()

        print('The accuracy of the best round is ', best_round)

    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator.reset()
        dataset = ChangeDetectionDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=8, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()

        for itera, data in enumerate(val_data_loader):
            pre_change_imgs, post_change_imgs, labels, _ = data
            pre_change_imgs = pre_change_imgs.cuda().float()
            post_change_imgs = post_change_imgs.cuda()
            labels = labels.cuda().long()

            output_1,_,_ = self.deep_model(pre_change_imgs, post_change_imgs)

            output_1 = output_1.data.cpu().numpy()
            output_1 = np.argmax(output_1, axis=1)
            labels = labels.cpu().numpy()

            self.evaluator.add_batch(labels, output_1)
        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        print(f'Racall rate is {rec}, Precision rate is {pre}, OA is {oa}, '
              f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')
        return rec, pre, oa, f1_score, iou, kc


def main():
    parser = argparse.ArgumentParser(description="Training on SYSU/LEVIR-CD+/WHU-CD dataset")
    parser.add_argument('--cfg', type=str, default='/media/mcislab/WYY/project/MambaCD/classification/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    #预训练模型路径
    parser.add_argument('--pretrained_weight_path', type=str,default="/media/mcislab/WYY/project/MambaCD/vssm_base_0229_ckpt_epoch_237.pth")
    parser.add_argument('--dataset', type=str, default='LEVIR-CD+')
    parser.add_argument('--type', type=str, default='train')
    #训练集路径
    #parser.add_argument('--train_dataset_path', type=str, default= "/media/mcislab/WYY/datasets/WHU-CD/train2/")
    parser.add_argument('--train_dataset_path', type=str,default="/media/mcislab/WYY/datasets/levircd/")
    #parser.add_argument('--train_dataset_path', type=str, default="/media/mcislab/WYY/datasets/Changen2")
    #训练集图像名称
    #parser.add_argument('--train_data_list_path', type=str, default="/media/mcislab/WYY/datasets/WHU-CD/train256/train_list.txt")
    parser.add_argument('--train_data_list_path', type=str,default="/media/mcislab/WYY/datasets/levircd/train_list.txt")
    #parser.add_argument('--train_data_list_path', type=str,default="/media/mcislab/WYY/datasets/Changen2/train_list.txt")
    #验证集路径
    parser.add_argument('--test_dataset_path', type=str, default='/media/mcislab/WYY/datasets/LEVIR-CD/val/')
    #parser.add_argument('--test_dataset_path', type=str, default='/media/mcislab/WYY/datasets/WHU-CD/val')
    #验证集路径名称
    parser.add_argument('--test_data_list_path', type=str, default='/media/mcislab/WYY/datasets/LEVIR-CD/val/val_list.txt')
    #parser.add_argument('--test_data_list_path', type=str, default='/media/mcislab/WYY/datasets/WHU-CD/val/val_list.txt')

    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=300000)
    parser.add_argument('--model_type', type=str, default='MambaBCD_FPN')

    #保存模型的路径
    parser.add_argument('--model_param_path', type=str, default='/media/mcislab/WYY/project/MambaCD/results/')

    #parser.add_argument('--resume', type=str,default="/media/mcislab/WYY/project/MambaCD/results/LEVIR-CD+/MambaBCD_FPN_202409141211/3330_model.pth")
    #微调的模型
    #parser.add_argument('--resume', type=str,default="/media/mcislab/WYY/project/MambaCD/results/LEVIR-CD+/MambaBCD_FPN_2025/WHU-CD-self/3-29-1122/6300_model.pth")
    parser.add_argument('--resume', type=str)

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()
    # wandb.init(
    #     project="ChangeDetection",  # 替换为你的项目名称
    #     config={
    #         "learning_rate": args.learning_rate,
    #         "batch_size": args.batch_size,
    #         "weight_decay": args.weight_decay,
    #         "max_iters": args.max_iters,
    #         "dataset": args.dataset,
    #         "model_type": args.model_type
    #     }
    # )
    with open(args.train_data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
