# 第五次实验：Whitted-Style Ray Tracing
兰江昕憬202411081061

## 实验简介

本实验基于 **Taichi** 实现经典 **Whitted-Style Ray Tracing（Whitted 风格光线追踪）**。

实验完成了课程要求的四个基础任务，并实现了一个选做功能（MSAA 抗锯齿）。

---

## 开发环境

| 环境 | 版本 |
|------|------|
| Python | 3.12 |
| Taichi | 1.7.4 |
| IDE | Visual Studio Code |

---

## 实现内容

### 任务一：搭建三维场景

完成了实验要求中的三维场景，包括：

- 无限大棋盘平面（Ground Plane）
- 红色漫反射球（Diffuse Sphere）
- 银色镜面球（Mirror Sphere）

---

### 任务二：实现基于迭代的光线弹射

由于 GPU 不适合递归，因此采用 **for 循环** 实现 Whitted 光线追踪。

主要实现：

- 主光线（Primary Ray）
- 镜面反射（Perfect Reflection）
- 最大弹射次数（Max Bounces）控制

---

### 任务三：实现硬阴影

通过向光源发射 **Shadow Ray** 判断遮挡关系，实现硬阴影。

同时采用

```python
Pnew = P + N * EPS
```

避免 Shadow Acne（自相交黑点）。

---

### 任务四：完成 UI 交互

使用 `ti.ui.Window` 创建交互界面。

支持实时调节：

- Light X
- Light Y
- Light Z
- Max Bounces
- Samples（MSAA）

---

### 选做：MSAA 抗锯齿

采用多重采样（MSAA）。

每个像素随机发射多条主光线，对颜色进行平均，从而减小物体边缘和棋盘格边缘的锯齿。

---

# 运行方法

安装依赖：

```bash
pip install -r requirements.txt
```

运行程序：

```bash
python main.py
```

---

# 实验演示

下面为程序运行录屏：

<p align="center">
    <img src="demo5.gif" width="900">
</p>

---

# 项目结构

```text
.
├── main.py
├── README.md
└── demo5.gif
```

---

# 实现效果

程序实现了：

- 棋盘平面
- 漫反射球
- 镜面球
- 镜面反射
- 硬阴影
- Shadow Acne 修复
- 光源实时调节
- 最大弹射次数调节
- MSAA 抗锯齿

当 **Max Bounces = 1** 时，仅计算一次光线弹射；

当 **Max Bounces > 1** 时，可以观察到镜面球中的多次反射效果。

---

# 参考

- Whitted, T. (1980). *An Improved Illumination Model for Shaded Display.*
- Taichi Programming Language
