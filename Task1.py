import os
import cv2
import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import graycomatrix, graycoprops

# ==========================================
# 0. 环境与路径配置
# ==========================================
# 获取当前脚本所在目录的绝对路径
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

TRAIN_DIR = os.path.join(current_dir, 'train')
TEST_DIR = os.path.join(current_dir, 'test')
TEST_CSV_PATH = os.path.join(current_dir, 'test.csv')
SUBMISSION_PATH = os.path.join(current_dir, 'submission.csv')

# 统一图像尺寸 (兼顾纹理细节和计算速度)
IMG_SIZE = 256

# 五类植物名称 (严格对应文件夹名)
CLASSES = ['Black-grass', 'Common wheat', 'Loose Silky-bent', 'Scentless Mayweed', 'Sugar beet']

# ==========================================
# 1. 图像处理工具 (预处理与增强)
# ==========================================
def create_mask_for_plant(image):
    """
    智能掩码生成：提取植物，去除碎石背景
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 调整 HSV 范围：
    # S > 50: 去除灰白色的石头
    # V > 50: 去除黑色阴影
    lower_hsv = np.array([30, 50, 50]) 
    upper_hsv = np.array([90, 255, 255])
    
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    
    # 形态学开运算 (Opening): 腐蚀后膨胀 -> 去除背景上细小的绿色噪点
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # 形态学闭运算 (Closing): 膨胀后腐蚀 -> 填补叶片内部的小孔洞
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    return mask

def augment_images(image):
    """
    数据增强：1张图 -> 6张图 (原图 + 翻转 + 旋转)
    """
    augmented = [image]
    # 翻转
    augmented.append(cv2.flip(image, 1)) # 水平
    augmented.append(cv2.flip(image, 0)) # 垂直
    
    # 旋转
    rows, cols = image.shape[:2]
    center = (cols / 2, rows / 2)
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # borderValue=(0,0,0) 填充黑色背景
        rotated = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        augmented.append(rotated)
        
    return augmented

# ==========================================
# 2. 特征提取核心 (Feature Engineering)
# ==========================================
def extract_geometric_features(mask):
    """
    几何形态特征：区分碎叶(Mayweed)和整叶(Beet)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return np.zeros(3)
    
    # 取最大轮廓
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area == 0: return np.zeros(3)

    # 1. Solidity (实心度): 面积 / 凸包面积
    # Mayweed 分叉多，凸包面积大，Solidity 低
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # 2. Aspect Ratio (长宽比)
    # 禾本植物通常细长
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # 3. Extent (矩形充满度)
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0
    
    return np.array([solidity, aspect_ratio, extent])

def extract_special_color_features(image, mask):
    """
    特殊颜色特征：检测 Black-grass 的紫色茎
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    total_plant_pixels = cv2.countNonZero(mask)
    
    if total_plant_pixels == 0:
        return np.array([0.0])
    
    # 定义紫色/红色范围 (Hue: 0-20 or 150-180) 且要有一定饱和度
    # OpenCV Hue范围是 0-180
    lower_purple1 = np.array([0, 40, 40])
    upper_purple1 = np.array([25, 255, 255])
    lower_purple2 = np.array([150, 40, 40])
    upper_purple2 = np.array([180, 255, 255])
    
    mask_p1 = cv2.inRange(image_hsv, lower_purple1, upper_purple1)
    mask_p2 = cv2.inRange(image_hsv, lower_purple2, upper_purple2)
    mask_purple = cv2.bitwise_or(mask_p1, mask_p2)
    
    # 确保只计算植物Mask内的紫色
    mask_purple = cv2.bitwise_and(mask_purple, mask)
    
    purple_ratio = cv2.countNonZero(mask_purple) / total_plant_pixels
    return np.array([purple_ratio])

def extract_features(image):
    # 1. 预处理
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    mask = create_mask_for_plant(image)
    segmented_img = cv2.bitwise_and(image, image, mask=mask)
    
    # --- A. 颜色直方图 (高分辨率 Hue) ---
    hsv = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2HSV)
    # Hue bins=60 (每3度一个bin)，捕捉细微颜色差异
    hist_hue = cv2.calcHist([hsv], [0], mask, [60], [0, 180])
    hist_sat = cv2.calcHist([hsv], [1], mask, [16], [0, 256])
    hist_val = cv2.calcHist([hsv], [2], mask, [16], [0, 256])
    cv2.normalize(hist_hue, hist_hue)
    cv2.normalize(hist_sat, hist_sat)
    cv2.normalize(hist_val, hist_val)
    
    # --- B. 特殊颜色特征 (紫色占比) ---
    purple_feat = extract_special_color_features(segmented_img, mask)
    
    # --- C. 纹理特征 (GLCM) ---
    gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    # 计算距离1的纹理
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    texture_feats = [graycoprops(glcm, prop).mean() for prop in props]
    
    # --- D. 形状特征 (Hu Moments) ---
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    # Log变换防止数值过小
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    # --- E. 几何形态特征 (Solidity等) ---
    geo_feats = extract_geometric_features(mask)
    
    # 拼接所有特征
    global_features = np.hstack([
        hist_hue.flatten(), 
        hist_sat.flatten(), 
        hist_val.flatten(),
        purple_feat,    # 1维
        texture_feats,  # 5维
        hu,             # 7维
        geo_feats       # 3维
    ])
    
    return global_features

# ==========================================
# 3. 主流程：数据读取与增强
# ==========================================
print(f"Step 1: 读取训练数据并增强 (Source: {TRAIN_DIR})")
if not os.path.exists(TRAIN_DIR):
    print("❌ 错误: 找不到 train 文件夹，请检查路径。")
    exit()

train_features = []
train_labels = []

for class_name in CLASSES:
    path = os.path.join(TRAIN_DIR, class_name)
    files = glob.glob(os.path.join(path, '*')) # 读取所有文件
    
    print(f"  - 正在处理: {class_name} ({len(files)} 张)")
    
    count = 0
    for file_path in files:
        img = cv2.imread(file_path)
        if img is None: continue
        
        # 数据增强 (1变6)
        aug_imgs = augment_images(img)
        
        for aug_img in aug_imgs:
            feats = extract_features(aug_img)
            train_features.append(feats)
            train_labels.append(class_name)
            count += 1
            
    print(f"    -> 增强后: {count} 张")

X = np.array(train_features)
y = np.array(train_labels)
print(f"特征矩阵形状: {X.shape}")

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ==========================================
# 4. 模型训练 (SVM + GridSearch)
# ==========================================
print("\nStep 2: 模型训练与调优...")

# 标准化 (SVM 必做)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练/验证 (分层抽样)
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# 参数网格 (C值调大一点，允许更复杂的边界)
param_grid = {
    'C': [1, 10, 100, 500],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']
}

# 5折交叉验证，优化 F1-Macro
grid = GridSearchCV(SVC(probability=True, class_weight='balanced'), 
                    param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)

grid.fit(X_train, y_train)

print(f"最佳参数: {grid.best_params_}")
print(f"最佳 CV F1-Macro: {grid.best_score_:.4f}")

# 验证集详细报告
y_pred = grid.predict(X_val)
print("\n验证集分类报告:")
print(classification_report(y_val, y_pred, target_names=le.classes_))
print("混淆矩阵:")
print(confusion_matrix(y_val, y_pred))

# ==========================================
# 5. 预测测试集 (生成 submission.csv)
# ==========================================
print("\nStep 3: 预测测试集...")

# 用全量数据(增强后)重新训练最佳模型
final_model = grid.best_estimator_
final_model.fit(X_scaled, y_encoded)

# 确定测试文件列表
test_files = []
if os.path.exists(TEST_CSV_PATH):
    df_test = pd.read_csv(TEST_CSV_PATH)
    # 检查列名，兼容 ID 或 file_name
    if 'ID' in df_test.columns:
        test_files = df_test['ID'].tolist()
    elif 'file_name' in df_test.columns:
        test_files = df_test['file_name'].tolist()
    else:
        print("⚠️ CSV中未找到 'ID' 列，将读取 test 文件夹全部图片。")
        test_files = os.listdir(TEST_DIR)
else:
    print("⚠️ 未找到 test.csv，读取 test 文件夹全部图片。")
    if os.path.exists(TEST_DIR):
        test_files = os.listdir(TEST_DIR)
    else:
        print("❌ 错误: test 文件夹不存在。")
        exit()

# 提取测试集特征
test_features_list = []
valid_ids = []

for file_name in test_files:
    img_path = os.path.join(TEST_DIR, file_name)
    img = cv2.imread(img_path)
    
    if img is not None:
        # 测试集不做增强，只提取原图
        feats = extract_features(img)
        test_features_list.append(feats)
        valid_ids.append(file_name)
    else:
        print(f"无法读取: {file_name}")
        # 填充0向量，保证行数一致 (可选)
        test_features_list.append(np.zeros(X.shape[1]))
        valid_ids.append(file_name)

# 预测
X_test = np.array(test_features_list)
X_test_scaled = scaler.transform(X_test) # 用训练集的 scaler
predictions = final_model.predict(X_test_scaled)
predicted_labels = le.inverse_transform(predictions)

# 保存
submission = pd.DataFrame({
    'ID': valid_ids,
    'Category': predicted_labels
})

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"\n✅ 任务完成！提交文件已保存至: {SUBMISSION_PATH}")
print(submission.head())