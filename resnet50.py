import os
import glob
import time
import tensorflow as tf
from imageai.Detection import ObjectDetection

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

start_time = time.time()
path = os.getcwd()
execution_path = (path + '\\night_images\\')
os.mkdir(path + '\\retest')
output_path = (path + '\\retest\\')
format = str('jpg')
print(execution_path)
filepath = glob.glob(execution_path + '*.' + format)

SLfilelist = []

n = 0
for i in filepath:
    SLfilelist.append(os.path.splitext(os.path.basename(filepath[n]))[0])
    n = n + 1
n = 0

for i in SLfilelist:
    oldpath = execution_path + SLfilelist[n] + '.' + str(format)
    newpath = output_path + 'new_' + SLfilelist[n] + '.' + str(format)
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(path , "yolo.h5")) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0
    detector.loadModel(detection_speed="flash")
    #print(os.path.join(execution_path , "1.jpg"))
    detections = detector.detectObjectsFromImage(input_image=oldpath, output_image_path=newpath)
    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("--------------------------------")
    n = n + 1

finish_time = time.time()
print(finish_time - start_time)