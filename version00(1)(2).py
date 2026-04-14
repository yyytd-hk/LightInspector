import sys
import os
import cv2
import numpy as np
import rasterio
import tkinter as tk
from tkinter import ttk, filedialog
from ultralytics import YOLO
import random

#准备阶段

#MODEL_PATH：训练的yolov8模型的存储地址
MODEL_PATH = "C:/Users/22963/yolov8-main/文件包(1)/文件包/best2.pt"

#CLASS_SCORE_MAP：不同类型的光源赋分
CLASS_SCORE_MAP = {
    "move light": 40,  # 动态光源
    "ad light": 80,  # 发光广告
    "stay light": 60,  # 静态光源
    "up light": 90  # 上射光
}


#mat_calc函数：输入经纬度，分数数据tif读取
def mat_calc(lon, lat):
    """
    纯 Python 实现：输入经纬度 → 返回分数 + 土地类型
    完全替代你的 MATLAB 函数
    :param lon: 经度
    :param lat: 纬度
    :return: score(分数), land_type(土地类型)
    """
    score_tif_path = r"C:/Users/22963/yolov8-main/文件包(1)/文件包/score_total.tif"
    lc_tif_path = r"C:/Users/22963/yolov8-main/文件包(1)/文件包/cover_hunan1.tif"

    # ===================== 1. 读取分数 TIFF =====================
    with rasterio.open(score_tif_path) as src_score:
        score_total = src_score.read(1)  # 读取第一个波段
        transform = src_score.transform  # 坐标映射参数

        # 经纬度 → 像素行列
        row, col = rasterio.transform.rowcol(transform, lon, lat)
        row = max(0, min(row, score_total.shape[0] - 1))
        col = max(0, min(col, score_total.shape[1] - 1))
        score = score_total[row, col]

    # ===================== 2. 读取土地类型 TIFF =====================
    with rasterio.open(lc_tif_path) as src_lc:
        LC = src_lc.read(1)
        height, width = LC.shape
        bounds = src_lc.bounds

        xmin = bounds.left
        xmax = bounds.right
        ymin = bounds.bottom
        ymax = bounds.top

        # 计算行列（和你 MATLAB 公式完全一致）
        col = int(round((lon - xmin) / (xmax - xmin) * width))
        row = int(round((ymax - lat) / (ymax - ymin) * height))

        # 防止越界
        col = max(0, min(col, width - 1))
        row = max(0, min(row, height - 1))
        val = LC[row, col]

    # ===================== 3. 土地类型映射 =====================
    # 有且区分城市与生态，后续评级分类别
    if val == 7:
        area_type = "城市区"
    else:
        area_type = "生态区"
    print(f"matlab分数：{score}")
    return score, area_type

#yolo识别框，亮度计算
def calculate_brightness_coeff(img_crop):
    """
    计算亮度比例系数k1（0~1）
    :param img_crop: BGR格式裁剪图像（YOLO识别出的光源框区域）
    :return: k1（亮度越高，k1越大，0~1，光污染越强）
    """
    # 1. 转灰度图，计算人眼感知亮度（cv2.COLOR_BGR2GRAY底层用BT.601标准，和全局计算统一）
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 2. 基础分：裁剪区域的平均亮度（保留原逻辑，保证兼容性）
    avg_brightness = np.mean(gray)
    score_base = avg_brightness / 255.0

    # 3. 【核心优化】眩光加权：统计框内过曝高亮像素占比，强化局部刺眼光源的影响
    # 阈值180：8位图像中，亮度>180定义为过曝眩光（可根据需求微调）
    glare_threshold = 180
    glare_mask = gray > glare_threshold
    glare_ratio = np.sum(glare_mask) / (gray.shape[0] * gray.shape[1])  # 0~1

    # 4. 融合策略：80%基础平均亮度 + 20%眩光占比
    # 既保留整体亮度，又放大局部过曝的影响，更符合光污染“刺眼即重污染”的定义
    k1 = 0.8 * score_base + 0.2 * glare_ratio
    # 不符合感官，因此乘以2.5
    k1 = k1 * 2.5
    # 5. 强制限定k1在0~1范围内，防止极端值
    k1 = np.clip(k1, 0.0, 1.0)
    return k1

#整张图片的亮光计算，输出该部分得分whole_light_score
def calculate_whole(img_rgb):
    # 1. 提取RGB三通道，计算人眼感知亮度（ITU-R BT.601标准）
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)
    brightness = 0.299 * r + 0.587 * g + 0.114 * b  # 人眼亮度公式

    # 2. 计算全局平均亮度，归一化到0~1
    avg_bright = np.mean(brightness)
    k0 = avg_bright / 255.0

    # 3. 【优化1】新增：高亮占比加权，让“刺眼眩光”的评分更突出
    # 只统计亮度>180的过曝像素（8位图像0-255，180是高亮度临界值，过滤暗部正常亮度）
    bright_mask = brightness > 180
    bright_ratio = np.sum(bright_mask) / (brightness.shape[0] * brightness.shape[1])

    # 4. 【优化2】加权融合：全局亮度(80%) + 高亮占比(20%)，更贴合人眼对光污染的感知
    k0 = 0.8 * k0 + 0.2 * bright_ratio

    # 5. 强制限定0~1范围
    k0 = np.clip(k0, 0.0, 1.0)
    whole_light_score = 10 * k0 * 2.5
    return whole_light_score


#整张图片蓝光计算，输出蓝光得分blue_score
def check_bluelight(img_rgb):
    # 1. 拆分 RGB 三个通道
    R = img_rgb[..., 0].astype(np.float32)
    G = img_rgb[..., 1].astype(np.float32)
    B = img_rgb[..., 2].astype(np.float32)

    # 2. 计算每个通道总和
    sum_R = np.sum(R)
    sum_G = np.sum(G)
    sum_B = np.sum(B)

    # 3. 总亮度（防止除0错误）
    total = sum_R + sum_G + sum_B
    if total < 1e-6:  # 图像全黑，无蓝光
        return 0.0

    # 4. 计算蓝光占比（0~1）
    blue_ratio = sum_B / total

    # 5. 映射到 0~100 原始分
    blue_raw_score = blue_ratio * 100

    # 6. 缩放到 0~7 分（照片模块固定上限）
    blue_score = blue_raw_score * (7 / 100)

    return blue_score


#image_path前端输入的照片，单张处理逻辑
def process_image(model, image_path):
    """
    处理单张图片，计算各类光源危险得分（分数越高越危险）
    :param model: YOLOv8模型实例
    :param image_path: 图片路径
    :return: pyscore总分,识别的光类型
    """
    # 读取图片（BGR格式）并转换为RGB
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片：{image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB用于亮度/蓝光计算

    # 模型推理（conf=0.25过滤低置信度框）
    results = model(img_rgb, conf=0.25)
    result = results[0]

    detected_list = []
    best_class = "无明显污染"

    if len(result.boxes) > 0:
        # 遍历所有检测框，收集全部类别
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]
            conf = float(box.conf[0])
            detected_list.append(class_name)

        # 拼接所有类别名（用顿号分隔，方便显示）
        # 这里修复你原来的报错！detected_list 已经是字符串，不需要 [0]
        best_class = "、".join(detected_list)

    # 初始化各类别亮度系数存储
    class_coeffs = {cls: {"k1_list": []} for cls in CLASS_SCORE_MAP.keys()}

    # 遍历所有检测框
    for box in result.boxes:
        # 获取类别名称
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        if cls_name not in CLASS_SCORE_MAP:
            continue  # 跳过非目标类别

        # 获取检测框坐标并防止越界
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_rgb.shape[1], x2)
        y2 = min(img_rgb.shape[0], y2)

        # 裁剪检测框区域并计算亮度系数（0~1）
        img_crop = img_rgb[y1:y2, x1:x2]
        k1 = calculate_brightness_coeff(img_crop)  # 仅计算亮度
        class_coeffs[cls_name]["k1_list"].append(k1)

    # ===================== ✅ 这里开始：修改为正确评分逻辑 =====================
    total_score = 0.0  # 实际得分总和
    total_max_possible = 0  # 动态最大总分（识别到几类加几类基础分）

    for cls_name, base_score in CLASS_SCORE_MAP.items():
        k1_list = class_coeffs[cls_name]["k1_list"]

        if not k1_list:
            continue  # 没识别 → 跳过，不加进来

        # 这个类别识别到了 → 算分，并把基础分加入动态最大总分
        avg_k1 = np.mean(k1_list)
        score = base_score * avg_k1
        total_score += score
        total_max_possible += base_score

    # 计算最终识别分数（0~18）
    if total_max_possible == 0:
        avg_4 = 0.0
    else:
        avg_4 = (total_score / total_max_possible) * 18  # 动态除数！

    avg_4 = round(avg_4, 2)

    # 计算整张图像的亮度分数和蓝光分数（0~1）
    whole_light_score = calculate_whole(img_rgb)
    blue_scocre = check_bluelight(img_rgb)

    # 用于检测，删除！
    pyscore = avg_4 + whole_light_score + blue_scocre
    print(f"蓝光分数：{blue_scocre}")
    print(f"整张分数：{whole_light_score}")
    print(detected_list)
    print(f"识别分数：{avg_4}")
    # ===================== 关键：返回 2 个值 =====================
    return round(pyscore, 2), detected_list

#评级模型（简陋版）
def grade_light_pollution(score, area_type):
    """
    根据亮度分数自动评级（结合国际标准）
    Args:
        score (float): 0~100 的归一化分数
        area_type (str): "city" 城市 / "ecology" 生态

    Returns:
        str: 等级名称
        str: 评价描述
    """
    # 确保分数在合法范围内
    score = max(0, min(100, score))

    if area_type == "城市区":
        # 引用 CIE 140 标准
        if score < 20:
            return "优"
        elif score < 40:
            return "良"
        elif score < 60:
            return "中"
        elif score < 80:
            return "较差"
        else:
            return "差"

    elif area_type == "生态区":
        # 引用 Bortle Scale 标准
        if score < 10:
            return "优"
        elif score < 25:
            return "良"
        elif score < 45:
            return "中"
        elif score < 70:
            return "较差"
        else:
            return "差"
    else:
        return "区域类型错误"

def tips(light_type):
    # 映射：光源类型 → 对应文件名
    file_map = {
        "stay light": "路灯.txt",
        "move light": "车灯.txt",
        "up light": "上射光.txt",
        "ad light": "广告牌.txt"
    }

    # 检查类型是否合法
    if light_type not in file_map:
        return ["暂无该类型光源的可行性建议"]

    file_name = file_map[light_type]
    try:
        # 读取文件所有行，去掉空行和换行
        with open(file_name, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # 核心修改：必取第一行，再从剩余行随机抽2条
        if not lines:  # 文件为空
            return ["暂无该类型光源的可行性建议"]

        # 1. 必选第一行
        first_line = lines[0]
        # 2. 剩余行
        remaining_lines = lines[1:]

        # 3. 从剩余行随机抽2条（不足2条就全返回）
        if len(remaining_lines) >= 2:
            selected = random.sample(remaining_lines, 2)
        else:
            selected = remaining_lines

        # 4. 最终结果：第一行 + 随机2条，共3条
        final_lines = [first_line] + selected
        return final_lines

    except FileNotFoundError:
        return [f"未找到 {file_name} 文件"]



def main():

    # ================== 前端输入 （大改动）==================
    lon = float(input("请输入经度："))
    lat = float(input("请输入纬度："))
    folder_path = input("请输入图片文件夹路径：")

   #分数部分1
    mat_score, area_type = mat_calc(lon, lat)

    # ========== Python逻辑（遍历图片/计算分数） ==========
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    scores = []
    pol = []

    #加入yolo模型
    model = YOLO(MODEL_PATH)

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # 只处理图片文件
        if os.path.splitext(file_name)[1].lower() in img_extensions:
            try:
                img_score,pol_type = process_image(model, file_path)
                pol.append(pol_type)
                scores.append(img_score)
            except Exception as e:
                print(f"警告：处理图片 {file_name} 失败！{e}")
                continue

    # ========== 3. 计算平均分（替换为你的逻辑） ==========

    if not scores:
        print("错误：无有效图片！")
        sys.exit(1)
    avg_score = sum(scores) / len(scores)
    n = len(scores)
    # 情况2：列表是样本（除以n-1）
    std_score = np.std(scores, ddof=1)
    # 综合计算公式
    py_score = avg_score * (1 + 0.3 * (std_score / 17.5))
    # 最终得分
    final_score = py_score + mat_score
    final_score = round(final_score,2)

    #评级
    grade = grade_light_pollution(final_score,area_type)

    #计算分数占比

    TYPES = ["move light", "ad light", "stay light", "up light"]

    # 2. 初始化字典，全部赋值 0
    count_dict = {t: 0 for t in TYPES}

    for item in pol:
        for key in item:
            if key in count_dict:  # 防止脏数据
                count_dict[key] += 1

    total = sum(count_dict.values())
    # 计算占比，覆盖原来的字典（变成 百分比 形式）
    # 处理 total=0 避免除0报错
    if total == 0:
        ratio_dict = {t: 0.0 for t in TYPES}
    else:
        ratio_dict = {key: round(val / total * 100, 2) for key, val in count_dict.items()}

    #几类光源的类型占比，存储在ratio_dict中，前端读取数据既可

    #提供意见
    sorted_items = sorted(ratio_dict.items(),key = lambda x:x[1], reverse=True)
    (key1, max1) = sorted_items[0]
    (key2, max2) = sorted_items[1]


    print(f"该地区为{area_type}")
    print(f"最终得分为{final_score}")
    print(f"最终评级为{grade}")
    print(f"各类型光源占比：{ratio_dict}")
    print(f"主要污染类型为{key1}、{key2}")
    print(f"可行性建议为：")
    tips1 = tips(key1)
    tips2 = tips(key2)
    print(tips1)
    print(tips2)

if __name__ == "__main__":
    main()



