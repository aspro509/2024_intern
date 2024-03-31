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

def split_image(img):
    # Get the width and height of the input image
    height, width = img.shape[:2]
    
    # Calculate the midpoint of the image
    midpoint = width // 2
    
    # Split the image into left and right halves
    left_half = img[:, :midpoint]
    right_half = img[:, midpoint:]
    
    return left_half, right_half

def 기간(date):
    priod = ["01_03","01_03","01_03", "04_06", "04_06", "04_06", "07_09","07_09","07_09", "10_12", "10_12", "10_12"]
    date = str(date)
    year = date[:-4]
    month = priod[int(date[-4:-2])-1]
    return year+month

def cutoff_1(img):
    return img[180:600, 40:-50]

def cutoff_2(img):
    thresh = cv.threshold(img, 245, 255, cv.THRESH_BINARY)[1]
        # Find the contours in the mask
    mask = np.array(thresh)
    dark_pixels = mask < 175

    coords = np.argwhere(dark_pixels)
    x0, y0 = coords.min(axis=0) 
    x1, y1 = coords.max(axis=0)
    

    crop_img = img[x0:x1, y0:y1]
    return crop_img

def cutoff_3(img):
    thresh = cv.threshold(img, 240, 255, cv.THRESH_BINARY)[1]
        # Find the contours in the mask
    mask = np.array(thresh)
    dark_pixels = mask < 175

    coords = np.argwhere(dark_pixels)
    x0, y0 = coords.min(axis=0) 
    x1, y1 = coords.max(axis=0)
    

    crop_img = img[x0:x1, y0:y1]
    return crop_img

text = cv.imread("./TMJ.png", cv.IMREAD_GRAYSCALE)
kernel = np.array(text)
def rm_text(img):
    res = cv.matchTemplate(img, kernel, cv.TM_CCOEFF_NORMED)
    
    _, _, _, maxloc = cv.minMaxLoc(res)
    img = cv.rectangle(img, (maxloc[0]-7, maxloc[1]-5), (maxloc[0]+100, maxloc[1]+20), 255, -1)
    return img
    

def padding(img):
    h, w = img.shape
    diff = abs(h-w)
    if h > w:
        return np.pad(img[:w+diff//2,:], ((0, 0), (diff//2-diff//4, diff//4) ), mode='maximum')
    else:
        return np.pad(img, ((diff-diff//2, diff//2), (0, 0)), mode='maximum')


count = 0
unmatched = []
train_X = []
train_X_56 = []
train_X_112 = []
train_X_448 = []
test_X = []
test_X_56 = []
test_X_112 = []
test_X_448 = []

train_y = []
test_y = []
sexTest = []
TMJ_t = []
ageTest = []



for i in df.index:
    data = df.loc[i]
    date = data["실시일자"]
    id = data["등록번호"]
    name = data["성명"]
    image_dir = "./face/" +기간(date) + "/" + str(id) +" "+ name + " 2.png"
    
    if not os.path.isfile(image_dir):
        count += 1
        #print(image_dir)
        #unmatched.append([date, str(id), name])
        continue
    img_array = np.fromfile(image_dir, np.uint8)
    image = cv.imdecode(img_array, cv.IMREAD_COLOR)

    # 이미지가 불러와졌나 검사
    if image is None:
        print(image_dir)
        continue


    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    right, left = split_image(cutoff_1(image))
    left = cutoff_3(rm_text(cutoff_2(left)[:-22,:]))
    left = cv.flip(left, 1)
    right = cutoff_3(rm_text(cutoff_2(right)[:-22,:]))

    left = padding(left)
    right = padding(right)

    left_56 = cv.resize(left, (56, 56), interpolation=cv.INTER_CUBIC)
    right_56 = cv.resize(right, (56, 56), interpolation=cv.INTER_CUBIC)

    left_112 = cv.resize(left, (112, 112), interpolation=cv.INTER_CUBIC)
    right_112 = cv.resize(right, (112, 112), interpolation=cv.INTER_CUBIC)

    left_224 = cv.resize(left, (224, 224), interpolation=cv.INTER_CUBIC)
    right_224 = cv.resize(right, (224, 224), interpolation=cv.INTER_CUBIC)

    left_448 = cv.resize(left, (448, 448), interpolation=cv.INTER_CUBIC)
    right_448 = cv.resize(right, (448, 448), interpolation=cv.INTER_CUBIC)

    if date > 20220631:
        cv.imwrite("./test/"+str(date)+"_"+str(id)+"_L.png", left)
        cv.imwrite("./test/"+str(date)+"_"+str(id)+"_R.png", right)
        if data["Rt. OA"] != 0.5:
            test_X.append(right_224)
            test_X_56.append(right_56)
            test_X_112.append(right_112)
            test_X_448.append(right_224)

            TMJ_t.append(data["Rt. 수치"])
            sexTest.append(data["성별"] == "남")
            ageTest.append(data["나이"])
        if data["Lt. OA"] != 0.5:
            test_X.append(left)
            test_y.append(int(data["Lt. OA"]))

            TMJ_t.append(data["Lt. 수치"])
            sexTest.append(data["성별"] == "남")
            ageTest.append(data["나이"])


    else:
        cv.imwrite("./train/"+str(date)+"_"+str(id)+"_L.png", left)
        cv.imwrite("./train/"+str(date)+"_"+str(id)+"_R.png", right)
        
        if data["Rt. OA"] != 0.5:
            train_X.append(right)
            train_y.append(int(data["Rt. OA"]))
        if data["Lt. OA"] != 0.5:
            train_X.append(left)
            train_y.append(int(data["Lt. OA"]))

    del img_array, image

test_X = np.array(test_X, np.uint8)
test_y = np.array(test_y)
TMJ_t = np.array(TMJ_t)
sexTest = np.array(sexTest)
ageTest = np.array(ageTest)
train_X = np.array(train_X, np.uint8)
train_y = np.array(train_y)

#np.save("train_y.npy", train_y)
#np.save("test_y.npy", test_y)
#np.save("Age_test.npy", ageTest)
#np.save("Sex_test.npy", sexTest)
#np.save("TMJ_test.npy", TMJ_t)
print(count)