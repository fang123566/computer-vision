# ASL 美国手语实时识别系统

MediaPipe 提取手部关键点 + SVM 分类，准确率 ~95%，浏览器实时展示。

---

## 项目结构

```
asl_web/
├── run.py              # 单入口：自动训练 + 启动服务
├── train_model.py      # 训练脚本（读CSV，SVM，保存model.pkl）
├── app.py              # Flask 后端（/predict /health）
├── templates/
│   └── index.html      # 前端（摄像头 + 骨架 + 字母展示）
├── install.bat         # Windows 一键安装依赖
└── requirements.txt    # Python 依赖清单
```

---

## 快速开始

### 第一步：安装依赖

双击 `install.bat`，或手动执行：
```bat
pip install flask mediapipe scikit-learn joblib opencv-python
```
> 如遇网络问题，关闭代理软件（Clash/V2Ray）后重试。

### 第二步：启动项目

```bat
python run.py
```

首次运行会自动训练模型（约 30–60 秒），之后直接启动服务器。

### 第三步：打开浏览器

访问 **http://localhost:5000**

---

## 可调参数

| 文件 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| `train_model.py` | `SAMPLES_PER_CLASS` | 200 | 每个字母取多少样本 |
| 浏览器页面 | 检测频率滑块 | 4 fps | 每秒识别次数 |

### 强制重新训练
```bat
python run.py --train
```

---

## 数据来源

`../Real-time-Vernacular-Sign-Language-Recognition-using-MediaPipe-and-Machine-Learning-master/Preprocessed-Data/american.csv`

- 原始数据：ASL A–Z 字母图片经 MediaPipe 提取的 42 个手部坐标
- 有效样本：~40,000 行（清除无手部检测的零行后）
