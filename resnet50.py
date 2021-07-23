import os
import cv2
import glob
import time
import numpy as np
import tensorflow as tf
from imageai.Detection import ObjectDetection


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

start_time = time.time()
path = os.getcwd()
execution_path = (path + '\\night_images\\')
os.mkdir(path + '\\boxed')
os.mkdir(path + '\\retest')
maskpath = (path + '\\boxed\\')
output_path = (path + '\\retest\\')
format = str('jpg')
#print(execution_path)
filepath = glob.glob(execution_path + '*.' + format)

SLfilelist = []

n = 0
for i in filepath:
    SLfilelist.append(os.path.splitext(os.path.basename(filepath[n]))[0])
    n = n + 1
n = 0
ans = []

for i in SLfilelist:
    #path
    oldpath = execution_path + SLfilelist[n] + '.' + str(format)
    secpath = maskpath + SLfilelist[n] + '.' + str(format)
    newpath = output_path + 'new_' + SLfilelist[n] + '.' + str(format)

    #detector
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(path , "yolo.h5")) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0
    detector.loadModel(detection_speed="flash")
    detections = detector.detectObjectsFromImage(input_image=oldpath, output_image_path=secpath)

    flag = 0
    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
        list = eachObject["box_points"]

        if((list[3] > 280 and list[0] > ((900 - 2 * list[3]) / 3)) or (list[3] > 280 and list[2] < ((2 * list[3] + 400) / 3))
            or (list[1] > 280 and list[2] < ((2 * list[1] + 400) / 3)) or (list[1] > 280 and list[0] > ((2 * list[1] + 400) / 3))):
            flag = flag + 1
        print("--------------------------------")
    n = n + 1
    if (flag != 0):
        ans.append(i)

    # ROI mask
    img = cv2.imread(secpath)
    b = np.array([[[300, 0], [0, 450], [650, 450], [350, 0]]], dtype=np.int32)
    im = np.zeros(img.shape[:2], dtype="uint8")
    cv2.polylines(im, b, 1, 255)
    cv2.fillPoly(im, b, 255)
    mask = im
    masked = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(newpath, masked)

finish_time = time.time()
print(finish_time - start_time)
print(len(ans))
print(ans)