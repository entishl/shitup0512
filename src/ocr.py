import ddddocr
import cv2

class OCR:
    def __init__(self):
        self.ocr = ddddocr.DdddOcr()
        print("OCR模型已加载。")

    def recognize(self, img_bgr):
        """
        识别图像中的数字
        :param img_bgr: BGR格式的numpy图像数组
        :return: 识别到的数字 (int)，如果识别失败或非数字则返回 None
        """
        if img_bgr.size == 0:
            return None

        # ddddocr 需要 bytes 输入，这里将 numpy array 编码为 png bytes
        _, img_bytes = cv2.imencode('.png', img_bgr)
        res = self.ocr.classification(img_bytes.tobytes())
        
        if res and res.isdigit():
            return int(res)
        return None
