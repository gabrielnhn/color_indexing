import numpy as np
import cv2 as cv
import os
import random
import math
import matplotlib.pyplot as plt
from datetime import datetime as dt

def random_color():
    return random.randrange(255),random.randrange(255),random.randrange(255)

def resize(image):
    RESIZE_SIZE = 9
    height, width, c = image.shape
    return cv.resize(image, (width//RESIZE_SIZE, height//RESIZE_SIZE), interpolation=cv.INTER_LINEAR)




objects_dir = "./new_objects"
objects = []
for obj_file in os.listdir(objects_dir):
    obj_file = os.path.join(objects_dir, obj_file)
    obj = {}
    obj["name"] = os.path.basename(obj_file)
    obj["image"] = resize(cv.imread(obj_file))
    objects.append(obj)
    # hsv = obj["image"]
    # hsv = cv.cvtColor(obj["image"],cv.COLOR_BGR2HSV)
    hsv = cv.cvtColor(obj["image"],cv.COLOR_BGR2HLS)
    # roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    # roihistH = cv.calcHist([hsv],[0], None, [180], [0, 256] )
    roihist = cv.calcHist([hsv],[0, 2], None, [256, 256], [0, 256, 0, 256] )
    # roihist = cv.calcHist([hsv],[0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256] )
    
    # cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
    obj["hist"] = roihist
    print(roihist)
    # plt.clean()
    # plt.plot(roihist[1, :], color="green")
    # plt.plot(roihistH, color="blue")
    # plt.title(obj["name"])
    # plt.show()


scenes_dir = "./new_scenes"

for scene_file in os.listdir(scenes_dir):
    scene_file = os.path.join(scenes_dir, scene_file)

    target = resize(cv.imread(scene_file))
    # hsvt = target
    # hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)
    hsvt = cv.cvtColor(target,cv.COLOR_BGR2HLS)

    for obj in objects:
        # dst = cv.calcBackProject([hsvt],[0],obj["hist"],[0,256],1)
        dst = cv.calcBackProject([hsvt],[0,2],obj["hist"],[0,256,0,256],1)
        # dst = cv.calcBackProject([hsvt],[0,1, 2],obj["hist"],[0,256,0,256, 0, 256],1)
        # Now convolute with circular disc
        disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        cv.filter2D(dst,-1,disc,dst)


        # threshold and binary AND
        _,thresh = cv.threshold(dst,16,256,cv.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv.erode(thresh, kernel, iterations=2)

        box = cv.boundingRect(thresh)
        box_is_empty = False

        gray = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
        inverted_mask = cv.bitwise_not(thresh)
        contours, _  = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        thresh = cv.merge((thresh,thresh,thresh))
        dst = cv.merge((dst,dst,dst))

        res = cv.bitwise_and(target,thresh)
        # res = cv.bitwise_and(gray,thresh)


        color = random_color()
        # cv.rectangle(target, (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3])), color, 2)
        # cv.putText(target,obj["name"],(int(box[0]), int(box[1]) + 20),cv.FONT_HERSHEY_PLAIN,3,color,3,)

        gray_background = cv.bitwise_and(gray, gray, mask=inverted_mask)

        # Grayscale 1 channel -> Grayscale 3 channels
        gray_background = np.stack((gray_background,)*3, axis=-1)

        # add highlighted forest to the grayscale background
        highlighted = cv.add(res, gray_background)

        cv.drawContours(highlighted, contours, -1, color, 1)

        h, _, _ = obj["image"].shape
        remain = target.shape[0] - h
        top = math.ceil(remain/2)
        bottom = math.floor(remain/2)

        if remain > 0:
            object_image = cv.copyMakeBorder(obj["image"], top, bottom, 0, 0, cv.BORDER_CONSTANT, None, color)
        else:
            object_image = obj["image"]
            target = cv.copyMakeBorder(target, -top, -bottom, 0, 0, cv.BORDER_CONSTANT, None, color)

        res = np.hstack((target,object_image,highlighted, dst))
        cv.imshow(obj["name"], res)
        if cv.waitKey(0) & 0xFF == ord('s'):
            cv.imwrite(f"{dt.now()}.png", res)

        cv.destroyAllWindows()

        # res = np.hstack((res, obj["image"]))
    # cv.imshow('res.jpg', target)
    # cv.waitKey(0)