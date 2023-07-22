import cv2
from vgg16 import My_VGG16
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# 打开摄像头
capture = cv2.VideoCapture(0)

# 调用封装人脸识别算法
path = r'D:\anaconda\envs\pytorch\lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
face_cas = cv2.CascadeClassifier(path)

# 设置device
device = torch.device("cuda:0")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 导入模型
model = My_VGG16()
model.load_state_dict(torch.load("emo12.pth"))
model = model.to(device)
model.eval()

# 设置数据处理
tran = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
])

# 标签列表
kinds = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

# 实现实时展示
while True:
    # 获取图片属性和图片列表
    ret, img = capture.read()

    # 复制对象，实现多窗口展示
    img2 = img.copy()

    # 多窗口展示

    cv2.namedWindow("real time", cv2.WINDOW_NORMAL)
    cv2.imshow('real time', img2)
    # cv2.destroyAllWindows()

    # 按q退出
    if cv2.waitKey(1) == ord('q'):
        break

    # 按e拍照
    if cv2.waitKey(1) == ord('e'):
        img3 = img.copy()
        # 调用函数获取脸部区域，识别到人脸返回坐标列表，或则返回空列表
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(gray)

        if len(faces) >= 1:
            for (x, y, w, h) in faces:
                # 画框
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # 由于模型输入是PIL图片对象，所以要转化
                # 调用模型
                roi_color = tran(Image.fromarray(np.uint8(img[y:y + h, x:x + w])))
                roi_color = torch.unsqueeze(roi_color, dim=0)
                roi_color = roi_color.to(device)
                prediction = model(roi_color)

                # 取出最可能标签
                predicted = torch.max(prediction.data, 1)[1]

                # 获取表情类型
                type = int(predicted.item())
                kind = kinds[type]
                cv2.putText(img, kind, (x, y - 7), 3, 1.2, (0, 255, 0), 2)
                cv2.namedWindow("expression", cv2.WINDOW_NORMAL)
                cv2.imshow('expression', img)
                cv2.namedWindow("screenshot", cv2.WINDOW_NORMAL)
                cv2.imshow('screenshot', img3)
