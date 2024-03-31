import pandas as pd
import numpy as np
import cv2 as cv
import os
import glob
import math
import random
import matplotlib.pyplot as plt


execel_file = "./bone_scan.xlsx"

df = pd.read_excel(execel_file, sheet_name="2020.01.01~2023.05.08", )


#split image to right side and left side.
#return lefthalf, righthalf


def 기간(date):
    priod = ["01_03","01_03","01_03", "04_06", "04_06", "04_06", "07_09","07_09","07_09", "10_12", "10_12", "10_12"]
    date = str(date)
    year = date[:-4]
    month = priod[int(date[-4:-2])-1]
    return year+month


def cutoff(img, thresholds):
    thresh = cv.threshold(img, thresholds, 255, cv.THRESH_BINARY)[1]
        # Find the contours in the mask
    mask = np.array(thresh)
    dark_pixels = mask < 175

    coords = np.argwhere(dark_pixels)
    x0, y0 = coords.min(axis=0) 
    x1, y1 = coords.max(axis=0)
    

    #crop_img = img[x0:x1, y0:y1]
    crop_img = img[x0:x1, :]
    return crop_img

def split_merge(image):

    _, w = image.shape
    img1 = np.full((700, 224, 2), 255, np.uint8)
    temp_img = np.full((800, 256, 2), 255, np.uint8)

    half_w = w // 2
    if w < 1100:
        img1[:,:224] = image[1:-1,17:half_w//2-16]
        img1[:,:,1] = cv.flip(image[1:-1,half_w//2+17:half_w//2+17+224], 1)
        #img1[:,:224,1] = image[1:-1,half_w+17:half_w+17+224]
        #img1[:,224:,1] = cv.flip(image[1:-1,w-half_w//2+17:-16], 1)
        return cv.resize(img1, (224, 352), interpolation=cv.INTER_CUBIC)
    else:
        temp_img[10:-10,: 1] = image[:,14:270]
        temp_img[10:-10,256:] = cv.flip(image[:,half_w//2+14:half_w//2+270], 1)
        #temp_img[10:-10,:256,1] = image[:,half_w+14:half_w+270]
        #temp_img[10:-10,256:,1] = cv.flip(image[:,w-half_w//2+14:w-half_w//2+270], 1)
        return cv.resize(temp_img, (224, 352), interpolation=cv.INTER_CUBIC)

text1 = cv.imread("./text1.png", cv.IMREAD_GRAYSCALE)
kernel1 = np.array(text1)
text2 = cv.imread("./text1.png", cv.IMREAD_GRAYSCALE)
kernel2 = np.array(text1)
def rm_text(img):
    res = cv.matchTemplate(img, kernel1, cv.TM_CCOEFF_NORMED)
    
    _, _, _, maxloc = cv.minMaxLoc(res)
    img[maxloc[1]-1:, :] = 255
    return img

count = 0
unmatched = []
train_X = []
test_X = []
train_y = []
test_y = []


W1 = []
W = []

for i in df.index:
    data = df.loc[i]
    date = data["실시일자"]
    id = data["등록번호"]
    name = data["성명"]
    image_dir = "./body/" +기간(date) + "/" + str(id) +" "+ name + " 1.png"


    
    if not os.path.isfile(image_dir):
        count += 1
        #print(image_dir)
        #unmatched.append([date, str(id), name])
        continue
    img_array = np.fromfile(image_dir, np.uint8)
    image = cv.imdecode(img_array, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 이미지가 불러와졌나 검사
    if image is None:
        print(image_dir)
        continue
    image = image[60:-20,:]
    try:
        img1 = split_merge(image)
    except ValueError:
        print(image_dir)
        continue


    #print(w)
    # cv.imshow("qwe", image)
    # cv.waitKey(0)
    if date > 20220631:
        cv.imwrite("./body_test/"+str(date)+"_"+str(id)+".png", img1[:,:])
        
        if data["Rt. OA"] == 1 or data["Lt. OA"] == 1:
            test_X.append(cv.cvtColor(img1, cv.COLOR_GRAY2BGR))
            test_y.append(1)
        elif data["Rt. OA"] == 0 and data["Lt. OA"] == 0:
            test_X.append(cv.cvtColor(img1, cv.COLOR_GRAY2BGR))
            test_y.append(0)
    else:
        cv.imwrite("./body_train/"+str(date)+"_"+str(id)+".png", img1[:,:])

        if data["Rt. OA"] == 1 or data["Lt. OA"] == 1:
            train_X.append(cv.cvtColor(img1, cv.COLOR_GRAY2BGR))
            train_y.append(1)
        elif data["Rt. OA"] == 0 and data["Lt. OA"] == 0:
            train_X.append(cv.cvtColor(img1, cv.COLOR_GRAY2BGR))
            train_y.append(0)
        

    del img_array, image

test_X = np.array(test_X, np.uint8)
test_y = np.array(test_y)
train_X = np.array(train_X, np.uint8)
train_y = np.array(train_y)

np.save("body_train_X.npy", train_X)
print(train_X.shape)
np.save("body_train_y.npy", train_y)
np.save("body_test_X.npy", test_X)
np.save("body_test_y.npy", test_y)