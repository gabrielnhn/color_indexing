import numpy as np
import cv2 as cv
import os
import random

def random_color():
    return random.randrange(255),random.randrange(255),random.randrange(255)

def resize(image):
    RESIZE_SIZE = 5
    height, width, c = image.shape
    return cv.resize(image, (width//RESIZE_SIZE, height//RESIZE_SIZE), interpolation=cv.INTER_LINEAR)




objects_dir = "./objects"
objects = []
for obj_file in os.listdir(objects_dir):
    obj_file = os.path.join(objects_dir, obj_file)
    obj = {}
    obj["name"] = os.path.basename(obj_file)
    obj["image"] = resize(cv.imread(obj_file))
    objects.append(obj)
    hsv = cv.cvtColor(obj["image"],cv.COLOR_BGR2HSV)
    roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
    obj["hist"] = roihist


scenes_dir = "./scenes"

for scene_file in os.listdir(scenes_dir):
    scene_file = os.path.join(scenes_dir, scene_file)
    target = resize(cv.imread(scene_file))
    hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)

    for obj in objects:
        dst = cv.calcBackProject([hsvt],[0,1],obj["hist"],[0,180,0,256],1)
        # Now convolute with circular disc
        disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        cv.filter2D(dst,-1,disc,dst)
        # threshold and binary AND
        ret,thresh = cv.threshold(dst,50,255,0)
        # thresh = cv.merge((thresh,thresh,thresh))
        # res = cv.bitwise_and(target,thresh)

        box = cv.boundingRect(thresh)
        color = random_color()
        cv.rectangle(target, (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3])), color, 2)
        cv.putText(target,obj["name"],(int(box[0]), int(box[1])),cv.FONT_HERSHEY_PLAIN,1,color,1,)

        # res = np.vstack((target,thresh,res))
        cv.imshow('res.jpg', target)
        cv.waitKey(0)