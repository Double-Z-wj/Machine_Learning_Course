import cv2
import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tqdm import tqdm
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ---------------------- 1. 全局配置 (请核对路径) ----------------------
CONFIG = {
    "img_size": 256,         # DenseNet 推荐输入尺寸
    "batch_size": 16,        # 显存不够可改为 8
    "epochs": 20,            # 每个 Fold 训练 20 轮 (增加几轮保证收敛)
    "n_folds": 5,            # 5折交叉验证
    "lr": 1e-4,              # 初始学习率
    "seed": 2024,            # 固定随机种子
    "num_workers": 0,        # Windows下建议设为0，避免多线程报错
    # 根据你报错信息填写的路径，无需修改
    "train_root": r"D:\deeplearn\dataset-for-task2\dataset-for-task2\train",
    "test_root": r"D:\deeplearn\dataset-for-task2\dataset-for-task2\test",
    "submission_path": r"D:\deeplearn\dataset-for-task2\dataset-for-task2\submission.csv"
}

# 植物类别 (按字母顺序排序，必须与文件夹顺序一致)
plant_classes = ['Black-grass', 'Common wheat', 'Loose Silky-bent', 'Scentless Mayweed', 'Sugar beet']

# ---------------------- 2. 基础工具函数 ----------------------
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CONFIG['seed'])

def segment_plant(image):
    """
    HSV 去背景算法：提取绿色植物，将背景置黑
    """
    # 转换到 HSV 空间
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # 绿色的 HSV 范围 (宽泛一点，避免切掉边缘)
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])
    
    # 生成掩膜
    mask = cv2.inRange(hsv_img, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 叠加掩膜
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

# ---------------------- 3. Dataset 定义 ----------------------
class PlantDataset(Dataset):
    def __init__(self, img_paths, labels=None, transform=None, is_train=True):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        # 读取图片
        img = cv2.imread(path)
        if img is None:
            # 容错处理
            img = np.zeros((CONFIG["img_size"], CONFIG["img_size"], 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 应用去背景 (这是提分关键)
        img = segment_plant(img)
        
        # 应用 PyTorch Transforms
        if self.transform:
            img = self.transform(img)
            
        if self.labels is not None:
            return img, self.labels[idx]
        else:
            # 测试集返回图片和文件名
            return img, os.path.basename(path)

# ---------------------- 4. 数据增强与 Mixup ----------------------
def get_transforms(data):
    if data == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((CONFIG["img_size"] + 20, CONFIG["img_size"] + 20)),
            transforms.RandomCrop(CONFIG["img_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif data == 'valid':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# Mixup 数据增强：将两张图按比例叠加
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ---------------------- 5. 模型定义 (DenseNet121) ----------------------
class PlantModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 自动下载预训练权重
        self.backbone = models.densenet121(weights='DEFAULT')
        
        # 修改分类层
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# ---------------------- 6. 训练与验证逻辑 (已修复报错) ----------------------
def train_fn(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0 # Mixup 下仅供参考
    total = 0
    
    # 进度条
    pbar = tqdm(loader, desc="Train", leave=False)
    
    for images, labels in pbar:
        # [FIX]: 强制转换为 long 类型，解决 RuntimeError
        images = images.to(device)
        labels = labels.to(device).long()
        
        # 应用 Mixup
        images, targets_a, targets_b, lam = mixup_data(images, labels)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # 计算 Loss
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # 简单统计准确率
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        # Mixup 下的准确率近似计算
        correct += (lam * preds.eq(targets_a).cpu().sum().float() + (1 - lam) * preds.eq(targets_b).cpu().sum().float())
        
        pbar.set_postfix({'loss': running_loss/total})
        
    return running_loss / total, correct / total

def valid_fn(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Valid", leave=False)
    
    with torch.no_grad():
        for images, labels in pbar:
            # [FIX]: 强制转换为 long 类型
            images = images.to(device)
            labels = labels.to(device).long()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += preds.eq(labels).cpu().sum().item()
            
            pbar.set_postfix({'acc': correct/total})
            
    return running_loss / total, correct / total

# ---------------------- 7. 主函数 ----------------------
def main():
    # 检查是否有 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 运行设备: {device}")
    if str(device) == 'cpu':
        print("⚠️ 警告: 未检测到 GPU，训练速度会非常慢！")

    # 1. 读取所有数据路径
    all_img_paths = []
    all_labels = []
    
    # 遍历训练集文件夹
    print("📂 正在读取数据...")
    for idx, cls_name in enumerate(plant_classes):
        cls_dir = os.path.join(CONFIG["train_root"], cls_name)
        if not os.path.exists(cls_dir):
            print(f"❌ 错误: 找不到文件夹 {cls_dir}")
            continue
            
        # 兼容多种图片格式
        paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            paths.extend(glob.glob(os.path.join(cls_dir, ext)))
            
        all_img_paths.extend(paths)
        all_labels.extend([idx] * len(paths))
        
    all_img_paths = np.array(all_img_paths)
    all_labels = np.array(all_labels)
    print(f"📊 总样本数: {len(all_img_paths)}")
    
    if len(all_img_paths) == 0:
        print("❌ 错误: 没有读取到任何图片，请检查 CONFIG['train_root'] 路径！")
        return

    # 2. 五折交叉验证循环
    skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["seed"])
    best_fold_scores = []
    
    # 存储模型文件名，方便后续加载
    model_files = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_img_paths, all_labels)):
        print(f"\n{'='*15} Fold {fold+1} / {CONFIG['n_folds']} {'='*15}")
        
        # 划分当前折的数据
        X_train, y_train = all_img_paths[train_idx], all_labels[train_idx]
        X_val, y_val = all_img_paths[val_idx], all_labels[val_idx]
        
        # 构建 Dataset 和 DataLoader
        train_ds = PlantDataset(X_train, y_train, transform=get_transforms('train'), is_train=True)
        val_ds = PlantDataset(X_val, y_val, transform=get_transforms('valid'), is_train=False)
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
        val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
        
        # 初始化模型、优化器
        model = PlantModel(len(plant_classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)
        
        best_acc = 0.0
        model_save_name = f"dense_fold_{fold}.pth"
        model_files.append(model_save_name)
        
        # 训练 Loop
        for epoch in range(CONFIG["epochs"]):
            train_loss, train_acc = train_fn(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = valid_fn(model, val_loader, criterion, device)
            scheduler.step()
            
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # 保存每一折的最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_save_name)
        
        print(f"🎉 Fold {fold+1} 最佳准确率: {best_acc:.4f}")
        best_fold_scores.append(best_acc)
        
    print(f"\n📈 所有 Fold 平均验证准确率: {np.mean(best_fold_scores):.4f}")

    # ---------------------- 8. 集成预测 (Ensemble Inference) ----------------------
    print("\n🚀 开始集成预测 (Ensemble Inference)...")
    
    # 准备测试数据
    test_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        test_paths.extend(glob.glob(os.path.join(CONFIG["test_root"], ext)))
        
    if len(test_paths) == 0:
        print("❌ 错误: 测试集为空，请检查路径！")
        return

    test_ds = PlantDataset(test_paths, transform=get_transforms('valid'), is_train=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"]*2, shuffle=False, num_workers=CONFIG["num_workers"])
    
    # 初始化总概率矩阵
    final_probs = np.zeros((len(test_paths), len(plant_classes)))
    
    # 遍历所有训练好的 5 个模型
    for fold in range(CONFIG["n_folds"]):
        print(f"正在加载 Fold {fold+1} 的模型进行预测...")
        model = PlantModel(len(plant_classes)).to(device)
        model.load_state_dict(torch.load(model_files[fold]))
        model.eval()
        
        fold_probs = []
        with torch.no_grad():
            for images, _ in tqdm(test_loader, leave=False):
                images = images.to(device)
                
                # TTA 策略: 预测原图 + 预测水平翻转图
                out1 = model(images)
                out2 = model(torch.flip(images, [3])) # [Batch, C, H, W]，dim=3 是宽度方向
                
                # 概率取平均
                probs = (torch.softmax(out1, 1) + torch.softmax(out2, 1)) / 2
                fold_probs.append(probs.cpu().numpy())
        
        # 累加当前模型的预测结果
        final_probs += np.concatenate(fold_probs)
        
    # 取 5 个模型的平均值
    final_probs /= CONFIG["n_folds"]
    
    # 获取最终类别
    predictions = np.argmax(final_probs, axis=1)
    pred_classes = [plant_classes[p] for p in predictions]
    img_names = [os.path.basename(p) for p in test_paths]
    
    # ---------------------- 9. 生成提交文件 ----------------------
    df = pd.DataFrame({'ID': img_names, 'Category': pred_classes})
    df.to_csv(CONFIG["submission_path"], index=False)
    
    print(f"\n✅ 预测完成！文件已保存至: {CONFIG['submission_path']}")
    print("预览前5行:")
    print(df.head())

if __name__ == "__main__":
    main()