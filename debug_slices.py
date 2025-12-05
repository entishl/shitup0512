import cv2
import numpy as np
import os
import ddddocr
from PIL import Image, ImageDraw, ImageFont
import PIL

# 兼容 Pillow v10.0.0+
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# ================= 配置区域 (从 main.py 复制) =================
# 游戏参数
GRID_ROWS = 16
GRID_COLS = 10

# 坐标参数 (基准分辨率 3840 * 2160)
START_X = 1455
START_Y = 488
STEP_X = 104
STEP_Y = 104

# 识别区域大小
CROP_SIZE = 50 

# 图像文件名
INPUT_IMAGE = "for_test.png"
OUTPUT_IMAGE = "debug_slices_output.png"
# ===========================================================

def draw_slice_boxes_with_ocr():
    """
    读取输入图片，对每个单元格的切片区域进行OCR，
    将区域用红框标出，并将识别结果用半透明红色字体标注在框内，
    最后保存到输出文件中。
    """
    # 1. 检查并读取输入图片
    if not os.path.exists(INPUT_IMAGE):
        print(f"错误: 输入图片 '{INPUT_IMAGE}' 不存在。")
        return

    img_cv = cv2.imread(INPUT_IMAGE)
    if img_cv is None:
        print(f"错误: 无法读取图片 '{INPUT_IMAGE}'。")
        return
    
    print(f"成功读取图片 '{INPUT_IMAGE}', 尺寸: {img_cv.shape[1]}x{img_cv.shape[0]}")

    # 初始化 OCR
    ocr = ddddocr.DdddOcr()
    print("DdddOcr 初始化完成。")

    # 2. 准备绘制：将 cv2 图像转为 Pillow 图像以便绘制透明文字
    # OpenCV 使用 BGR, Pillow 使用 RGB
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(img_pil)

    # 准备字体
    try:
        # 尝试加载一个常见的字体，例如微软雅黑
        font = ImageFont.truetype("msyh.ttc", 30)
    except IOError:
        print("警告: 找不到 'msyh.ttc' 字体, 将使用默认字体。")
        font = ImageFont.load_default()

    # 定义颜色
    RED_COLOR_BOX = (255, 0, 0, 255)  # 纯红框 (R, G, B, A)
    RED_COLOR_TEXT = (255, 0, 0, 180) # 半透明红字 (R, G, B, A)
    THICKNESS = 2

    # 3. 遍历网格，执行 OCR 并绘制
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            # 计算每个单元格中心点的绝对屏幕坐标
            center_x = START_X + c * STEP_X
            center_y = START_Y + r * STEP_Y
            
            # 计算切片区域的左上角和右下角坐标
            half_crop = CROP_SIZE // 2
            x1 = center_x - half_crop
            y1 = center_y - half_crop
            x2 = center_x + half_crop
            y2 = center_y + half_crop
            
            # 绘制矩形框
            draw.rectangle([(x1, y1), (x2, y2)], outline=RED_COLOR_BOX, width=THICKNESS)

            # 从原始 cv2 图像中提取切片用于 OCR
            cell_img_cv = img_cv[y1:y2, x1:x2]
            
            # 执行 OCR
            _, img_bytes = cv2.imencode('.png', cell_img_cv)
            res = ocr.classification(img_bytes.tobytes())
            
            # 如果识别结果是数字，则绘制文本
            if res and res.isdigit():
                # 获取文本尺寸以进行居中
                try:
                    # Pillow >= 9.2.0
                    bbox = draw.textbbox((0, 0), res, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError:
                    # Pillow < 9.2.0
                    text_width, text_height = draw.textsize(res, font=font)

                # 计算文本位置使其在框内居中
                text_x = x1 + (CROP_SIZE - text_width) / 2
                text_y = y1 + (CROP_SIZE - text_height) / 2
                
                # 绘制半透明文本
                draw.text((text_x, text_y), res, font=font, fill=RED_COLOR_TEXT)

    # 4. 保存结果图片
    # 将 Pillow 图像转回 cv2 格式以便保存
    img_final_cv = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUTPUT_IMAGE, img_final_cv)
    print(f"处理完成，结果已保存到 '{OUTPUT_IMAGE}'")

if __name__ == "__main__":
    draw_slice_boxes_with_ocr()