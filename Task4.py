import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# -------------------------- 0. 自动路径 --------------------------
def find_dataset_path():
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'fovea_localization_train_GT.csv' in files: return root
    return None

base_path = find_dataset_path()
train_img_path = os.path.join(base_path, 'train')
test_img_path = os.path.join(base_path, 'test')
train_csv_path = os.path.join(base_path, 'fovea_localization_train_GT.csv')
sample_submission_path = os.path.join(base_path, 'sample_submission.csv')

# -------------------------- 1. 参数 (严格对齐 14 分成功版本) --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256          # 严格回到 256，这是你成功的关键
BATCH_SIZE = 8
EPOCHS = 70             # 稍微多跑几轮确保收敛
LEARNING_RATE = 1e-4
GAUSSIAN_SIGMA = 5      # 保持 5 

# -------------------------- 2. 原始 UNet (14 分版本的模型基因) --------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(1, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.inc(x); e2 = self.down1(e1); e3 = self.down2(e2); e4 = self.down3(e3); e5 = self.down4(e4)
        d1 = self.up1(e5); d1 = self.dec1(torch.cat([d1, e4], 1))
        d2 = self.up2(d1); d2 = self.dec2(torch.cat([d2, e3], 1))
        d3 = self.up3(d2); d3 = self.dec3(torch.cat([d3, e2], 1))
        d4 = self.up4(d3); d4 = self.dec4(torch.cat([d4, e1], 1))
        return self.final(d4)

# -------------------------- 3. 破 13 分的核弹：极窄重心微调 --------------------------
def get_final_subpixel_coord(heatmap):
    """
    14 分版本用的是整数 Argmax。
    我们要破 13 分，必须通过这个重心法拿到小数坐标。
    只取 3x3 邻域，这能极大避免噪声拉偏重心，只做像素级的微调。
    """
    heatmap = heatmap.squeeze()
    y_m, x_m = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    # 取 3x3 窗口 (这是最稳的提分半径)
    r = 1 
    y1, y2 = max(0, y_m-r), min(IMG_SIZE, y_m+r+1)
    x1, x2 = max(0, x_m-r), min(IMG_SIZE, x_m+r+1)
    roi = heatmap[y1:y2, x1:x2]
    
    # 算重心
    s = np.sum(roi)
    if s == 0: return float(x_m), float(y_m)
    iy, ix = np.indices(roi.shape)
    return x1 + np.sum(ix * roi)/s, y1 + np.sum(iy * roi)/s

# -------------------------- 4. 数据处理 (严格灰度版) --------------------------
class FoveaDataset(Dataset):
    def __init__(self, paths, coords=None, transform=None, is_test=False):
        self.paths, self.coords, self.transform, self.is_test = paths, coords, transform, is_test
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        # 强制灰度图读入，保持 14 分版本的稳定性
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        h_o, w_o = img.shape[:2]
        img_res = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        if self.is_test:
            if self.transform: img_res = self.transform(img_res)
            return img_res, os.path.basename(self.paths[idx]).split('.')[0], (h_o, w_o)
        
        x, y = self.coords[idx]
        x_s, y_s = x*(IMG_SIZE/w_o), y*(IMG_SIZE/h_o)
        xx, yy = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE))
        hm = np.exp(-((xx-x_s)**2 + (yy-y_s)**2)/(2*GAUSSIAN_SIGMA**2))
        if self.transform: img_res = self.transform(img_res)
        return img_res, torch.from_numpy(hm).float().unsqueeze(0)

# -------------------------- 5. 训练与 TTA 融合 --------------------------
def main():
    df = pd.read_csv(train_csv_path)
    img_paths = [os.path.join(train_img_path, f"{int(row['data']):04d}.jpg") for _, row in df.iterrows()]
    coords = [(row['Fovea_X'], row['Fovea_Y']) for _, row in df.iterrows()]
    
    # 尽可能给模型训练数据 (95% 训练)
    t_p, v_p, t_c, v_c = train_test_split(img_paths, coords, test_size=0.05, random_state=42)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_loader = DataLoader(FoveaDataset(t_p, t_c, transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FoveaDataset(v_p, v_c, transform), batch_size=BATCH_SIZE)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    # 动态学习率，后期细化位置
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    print("🚀 正在以 256px 灰度稳健版冲击 13 分以内...")
    best_v = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        for imgs, hms in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(imgs.to(device)), hms.to(device))
            loss.backward(); optimizer.step()
        
        model.eval()
        v_l = 0
        with torch.no_grad():
            for imgs, hms in val_loader:
                v_l += criterion(model(imgs.to(device)), hms.to(device)).item()
        
        scheduler.step(v_l)
        if v_l < best_v:
            best_v = v_l
            torch.save(model.state_dict(), 'best_stable_256.pth')
            print(f"🌟 Epoch {epoch+1} Saved")

    # 预测阶段：双路 TTA 融合 (降分利器)
    model.load_state_dict(torch.load('best_stable_256.pth'))
    model.eval()
    test_files = [os.path.join(test_img_path, f"{i:04d}.jpg") for i in range(81, 101)]
    test_loader = DataLoader(FoveaDataset([f for f in test_files if os.path.exists(f)], is_test=True, transform=transform), batch_size=1)
    
    results = {}
    with torch.no_grad():
        for img, img_id, (h_o, w_o) in test_loader:
            img = img.to(device)
            # 原图热力图
            p1 = model(img).squeeze().cpu().numpy()
            # 镜像图热力图并翻转回来
            p2 = np.fliplr(model(torch.flip(img, [3])).squeeze().cpu().numpy())
            
            # 融合热力图能极大抵消单次预测偏差
            avg_hm = (p1 + p2) / 2
            x_s, y_s = get_final_subpixel_coord(avg_hm)
            
            results[int(img_id[0])] = (x_s * (w_o.item()/IMG_SIZE), y_s * (h_o.item()/IMG_SIZE))

    sub_df = pd.read_csv(sample_submission_path)
    for i_id, (x, y) in results.items():
        sub_df.loc[sub_df['ImageID'] == f"{i_id}_Fovea_X", 'value'] = x
        sub_df.loc[sub_df['ImageID'] == f"{i_id}_Fovea_Y", 'value'] = y
    sub_df.to_csv('submission_sprint_v13_stable.csv', index=False)
    print("✅ 任务完成，请提交 submission_sprint_v13_stable.csv")

if __name__ == "__main__":
    main()