import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import time

class CarlaDataset(Dataset):
    def __init__(self, root_dir, split, image_transform, label_transform):
        self.split_dir = os.path.join(root_dir, split)
        self.img_dir = os.path.join(self.split_dir, "RGB")
        self.label_dir = os.path.join(self.split_dir, "Label")
        
        self.images = sorted(os.listdir(self.img_dir))
        self.labels = sorted(os.listdir(self.label_dir))
        
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label_name = self.labels[idx]
        
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)
        
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L") 

        image = self.image_transform(image)
        
        # 라벨(PIL)을 먼저 리사이즈
        label = self.label_transform(label) 
        
        # 리사이즈된 라벨을 텐서로 변환
        label_np = np.array(label, dtype=np.int64) 
        label_np[label_np == 255] = 0
        label = torch.from_numpy(label_np)

        return image, label

# --- 3. 모델 아키텍처 (수정됨) ---
class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SimpleUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 인코더 (Down-sampling)
        self.inc = DoubleConv(n_channels, 64)
        
        # --- ⚠️ 수정된 부분 1: 튜플 대신 개별 속성으로 등록 ---
        self.pool = nn.MaxPool2d(2)
        self.down_conv1 = DoubleConv(64, 128)
        self.down_conv2 = DoubleConv(128, 256)
        self.down_conv3 = DoubleConv(256, 512)
        # -----------------------------------------------

        # 디코더 (Up-sampling)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256) # Skip connection 포함
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        # 최종 출력 레이어
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # --- ⚠️ 수정된 부분 2: forward 로직 변경 ---
        # 인코더
        x1 = self.inc(x)
        
        x_pooled = self.pool(x1)
        x2 = self.down_conv1(x_pooled)
        
        x_pooled = self.pool(x2)
        x3 = self.down_conv2(x_pooled)
        
        x_pooled = self.pool(x3)
        x4 = self.down_conv3(x_pooled)
        # ------------------------------------------

        # 디코더 (Skip Connection 사용)
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1) # Skip connection
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        
        # 최종 출력
        logits = self.outc(x) # (N, C, H, W)
        return logits