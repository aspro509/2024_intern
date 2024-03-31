import pandas as pd
import numpy as np
import cv2 as cv
import os
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
    img1 = np.full((702, 224, 1), 255, np.uint8)
    temp_img = np.full((740, 224, 1), 255, np.uint8)

    half_w = w // 2
    if w < 1100:
        img1[:,:,0] = image[94:,17:224]
        #img1[:,:,1] = cv.flip(image[:,half_w//2+17:half_w//2+17+224], 1)
        return img1
    else:
        temp_img[:,:,0] = image[118:,30:254]
        #temp_img[:,:,1] = cv.flip(image[:,half_w//2+14:half_w//2+270], 1)
        return temp_img


count = 0
unmatched = []
train_X1 = []
train_X2 = []
test_X1 = []
test_X2 = []
train_y = []
test_y = []

train_sex = []
train_age = []
test_sex = []
test_age = []




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
    h, w = image.shape
    if w == 1132:
        image = cv.resize(image[65:-19,:], (1030, 706), interpolation=cv.INTER_CUBIC)
    elif w == 1030:
        image = image[60:-16,:]
    else:
        print(image_dir, "implict img size")
        continue

    img1 = image[98:, 17:17+224]
    img2 = image[98:, 274:274+224]


    if date > 20220631:
        cv.imwrite("./body_test/"+str(date)+"_"+str(id)+".png", img2)
        
        if data["Rt. OA"] == 1 or data["Lt. OA"] == 1:
            test_X1.append(cv.cvtColor(img1, cv.COLOR_GRAY2BGR))
            test_X2.append(cv.cvtColor(img2, cv.COLOR_GRAY2BGR))
            test_y.append(1)
            test_age.append(data["나이"])
            test_sex.append(data["성별"]=='남')
        elif data["Rt. OA"] == 0 and data["Lt. OA"] == 0:
            test_X1.append(cv.cvtColor(img1, cv.COLOR_GRAY2BGR))
            test_X2.append(cv.cvtColor(img2, cv.COLOR_GRAY2BGR))
            test_y.append(0)
            test_age.append(data["나이"])
            test_sex.append(data["성별"]=='남')
    else:
        cv.imwrite("./body_train/"+str(date)+"_"+str(id)+".png", img1)

        if data["Rt. OA"] == 1 or data["Lt. OA"] == 1:
            train_X1.append(cv.cvtColor(img1, cv.COLOR_GRAY2BGR))
            train_X2.append(cv.cvtColor(img2, cv.COLOR_GRAY2BGR))
            train_y.append(1)
            train_age.append(data["나이"])
            train_sex.append(data["성별"]=='남')
        elif data["Rt. OA"] == 0 and data["Lt. OA"] == 0:
            train_X1.append(cv.cvtColor(img1, cv.COLOR_GRAY2BGR))
            train_X2.append(cv.cvtColor(img2, cv.COLOR_GRAY2BGR))
            train_y.append(0)
            train_age.append(data["나이"])
            train_sex.append(data["성별"]=='남')
        

    del img_array, image

test_X1 = np.array(test_X1, np.uint8)
test_X2 = np.array(test_X2, np.uint8)
test_y = np.array(test_y)
train_X1 = np.array(train_X1, np.uint8)
train_X2 = np.array(train_X2, np.uint8)
train_y = np.array(train_y)
train_age = np.array(train_age)
train_sex = np.array(train_sex)
test_age = np.array(test_age)
test_sex = np.array(test_sex)



# np.save("body_train_X1.npy", train_X1)
# np.save("body_train_X2.npy", train_X2)
# np.save("body_train_y.npy", train_y)
# np.save("body_test_X1.npy", test_X1)
# np.save("body_test_X2.npy", test_X2)
# np.save("body_test_y.npy", test_y)

np.save("body_age_train.npy", train_age)
np.save("body_sex_train.npy", train_sex)
np.save("body_age_test.npy", test_age)
np.save("body_sex_test.npy", test_sex)