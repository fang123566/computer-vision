# ASL 美国手语实时识别系统

基于 **MediaPipe HandLandmarker + SVM** 的实时 A–Z 手语识别，测试准确率 **99.65%**，浏览器实时展示骨架与预测结果。

---

## 技术方案

| 模块 | 说明 |
|------|------|
| 手部检测 | MediaPipe HandLandmarker（Tasks API），IMAGE 模式 |
| 特征工程 | 21 个关键点 (x, y) → **归一化特征**（手腕为原点，手掌大小为单位），消除位置/尺寸影响 |
| 分类器 | SVM（RBF 核，C=100，gamma=0.1） |
| 图像预处理 | CLAHE 自适应对比度增强，改善弱光/过曝环境 |
| 手部选择 | 多手时按 reach × handedness_score 评分，过滤虚假检测 |
| 前端 | 原生 JS + Canvas 实时绘制骨架，显示 Top-3 预测与置信度 |

---

## 项目结构

```
asl_web/
├── run.py                  # 单入口：自动训练（如缺少model.pkl）+ 启动服务
├── app.py                  # Flask 后端（/predict  /health）
├── train_model.py          # 训练脚本：归一化特征 → SVM → model.pkl
├── extract_from_images.py  # 从 ASL_Dataset.zip 直接提取关键点特征
├── model.pkl               # 已训练模型（可直接使用）
├── hand_landmarker.task    # MediaPipe 手部检测模型
├── data/
│   ├── american.csv        # 公开数据集预提取特征（~40,000 行）
│   └── user_dataset.csv    # 从 ASL_Dataset.zip 提取的特征（300张×26类）
├── templates/
│   └── index.html          # 前端页面
├── install.bat             # Windows 一键安装依赖
└── requirements.txt        # Python 依赖清单
```

---

## 快速开始

### 第一步：安装依赖

双击 `install.bat`，或手动执行：
```bat
cd asl_web
pip install -r requirements.txt
```

> 如遇网络问题，关闭代理软件（Clash / V2Ray）后重试，或使用镜像源：
> `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 第二步：启动项目

```bat
python run.py
```

`model.pkl` 已包含在仓库中，**无需重新训练即可直接运行**。

### 第三步：打开浏览器

访问 **http://localhost:5000**

---

## 使用说明

- 将手放在摄像头前，字母骨架点自动跟踪
- 右侧显示当前预测字母、置信度，以及 Top-3 候选
- 持续保持某个手势可自动追加到拼写记录
- 可调节页面底部的**检测频率**滑块（建议 4–8 fps）

> **最佳识别条件**：手部处于画面中央，背景简洁，光线均匀，避免背后强光源。

---

## 可调参数

| 文件 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| `train_model.py` | `SAMPLES_PER_CLASS` | 200 | american.csv 每类取样数 |
| `extract_from_images.py` | `MAX_PER_CLASS` | 300 | 从 zip 提取的每类图片数 |
| `app.py` | `min_hand_detection_confidence` | 0.5 | MediaPipe 检测阈值 |
| 浏览器页面 | 检测频率滑块 | 4 fps | 每秒识别帧数 |

---

## 重新训练

如需基于更多数据重新训练：

```bat
# 1. 从 ASL_Dataset.zip 重新提取关键点（需要 zip 文件在 Downloads 目录）
python extract_from_images.py

# 2. 重新训练并启动
python run.py --train
```

---

## 数据来源

| 数据 | 说明 |
|------|------|
| `data/american.csv` | 公开 ASL 数据集预提取的 42 维手部坐标，~40,000 行 |
| `data/user_dataset.csv` | 从 `ASL_Dataset.zip` 直接读取图片，MediaPipe 提取关键点，300张×26类 = 7,800 行 |

训练时两份数据合并（user_dataset 重复 3× 提高权重），统一做**手腕坐标归一化**后送入 SVM。
