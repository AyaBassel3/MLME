import cv2
import numpy as np
import os
import cv2 as cv
from numpy import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import interpolation
def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix

def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)


def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img

def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def horizontal_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift)]
    if ratio < 0:
        img = img[:, int(-1*to_shift):]
    img = fill(img, h, w)
    return img

def vertical_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :]
    if ratio < 0:
        img = img[int(-1*to_shift):, :]
    img = fill(img, h, w)
    return img

def read_dataset(data_type,dim):
    img_count=0
    data_set=[]
    labels=[]
    root=os.path.join('D:/Masters/TU Dortmund/2nd Semester/Machine Learning/Project 3 materials/project_files/data/', data_type)


    for participant in os.listdir(root):
        participant_dir = os.path.join(root, participant)
        if os.path.isfile(participant_dir):
            continue
        for symbol in os.listdir(participant_dir):
            symbol_dir=os.path.join(participant_dir,symbol)
            if os.path.isfile(symbol_dir):
                continue
            for img in os.listdir(symbol_dir):
                img_path=os.path.join(symbol_dir, img)
                img=cv2.imread(img_path)
                gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                kernel_aya = np.ones((5, 5), np.float32) / 25
                #imgray = (255 - (cv2.filter2D(gray, -1, kernel_aya)))  # avg filter
                imgray = 255-(cv2.GaussianBlur(gray, (5, 5), 0))

                #imgray = 255-(cv.bilateralFilter(gray,9,70,70))

                ######################################
                imgray=deskew(imgray)


                #####################################
                canny_output = cv.Canny(imgray, 20, 200)
                canny_output = cv2.adaptiveThreshold(canny_output, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,2)
                contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                height = canny_output.shape[0]
                width = canny_output.shape[1]
                min_x, min_y = width, height
                max_x = max_y = 0

                for cnt in contours:
                    x, y, w, h = cv.boundingRect(cnt)
                    min_x, max_x = min(x, min_x), max(x + w, max_x)
                    min_y, max_y = min(y, min_y), max(y + h, max_y)

                rt=0.1
                if max_x - min_x > 0 and max_y - min_y > 0:
                    if (max_x - min_x) > (max_y - min_y):
                        mid_y = min_y + ((max_y - min_y) / 2)
                        tolerance = int(rt*(max_x-min_x)) + 1
                        dst = imgray[int(mid_y - ((max_x - min_x) / 2)) - tolerance if int(mid_y - ((max_x - min_x) / 2)) - tolerance >0 else 0: int(mid_y + ((max_x - min_x) / 2)) + tolerance,
                                  min_x - tolerance if min_x - tolerance > 0 else 0 : max_x + tolerance]
                    else:
                        tolerance = int(rt * (max_y - min_y)) + 1
                        mid_x = min_x + ((max_x - min_x) / 2)
                        dst = imgray[min_y - tolerance if min_y - tolerance >0 else 0 : max_y + tolerance,
                                  int(mid_x - ((max_y - min_y) / 2)) - tolerance if int(mid_x - ((max_y - min_y) / 2)) - tolerance > 0 else 0  : int(mid_x + ((max_y - min_y) / 2)) + tolerance]
                else:
                    dst= (255-(cv2.filter2D(gray, -1, kernel_aya)))
                ########################################################
                dst=cv2.GaussianBlur(dst, (11, 11), 0)

                (thresh, blackAndWhiteImage) = cv2.threshold(dst, 20, 255, cv2.THRESH_BINARY)

                rv=0.35
                rh=0.35
                height = dst.shape[0]
                width = dst.shape[1]
                paddingVertical= int(height * rv) + 1
                paddingHorizantal = int(width * rh) + 1

                Im_padded = cv2.copyMakeBorder(blackAndWhiteImage, paddingVertical , paddingVertical, paddingHorizantal , paddingHorizantal, cv2.BORDER_CONSTANT, None, value=0)


                res = cv2.resize(Im_padded, dsize=(dim, dim), interpolation=cv2.INTER_CUBIC)




                img_count=img_count+1
                lbl_out = np.zeros(10)
                lbl_out[int(symbol)] = 1
                print(img_count)
                data_set.append(res / np.max(res))
                #labels.append(lbl_out) #hot encoded for NN
                labels.append(symbol) #regular for SVM/KNN
                x=False

                if (data_type=='train' and x == True):
                    # rotation
                    img_rotation=rotation(res,30)
                    img_count = img_count + 1
                    lbl_out = np.zeros(10)
                    lbl_out[int(symbol)] = 1
                    print(img_count)
                    data_set.append( img_rotation / np.max(img_rotation))
                    #labels.append(lbl_out) #hot Encoded
                    labels.append(symbol) #regular

                    # vertical shift
                    img_VerticalShift = vertical_shift(res, 0.2)
                    img_count = img_count + 1
                    lbl_out = np.zeros(10)
                    lbl_out[int(symbol)] = 1
                    print(img_count)
                    data_set.append(img_VerticalShift / np.max(img_VerticalShift))
                    #labels.append(lbl_out) #hot Encoded
                    labels.append(symbol)  # regular

                    # horizontal shift
                    img_HorizontalShift = horizontal_shift(res, 0.2)
                    img_count = img_count + 1
                    lbl_out = np.zeros(10)
                    lbl_out[int(symbol)] = 1
                    print(img_count)
                    data_set.append(img_HorizontalShift / np.max(img_HorizontalShift))
                    #labels.append(lbl_out) #hot Encoded
                    labels.append(symbol)  # regular













    data=np.array(data_set) #np array ashan feeh features ktiir
    labels=np.array(labels)
    np.save('data_set',data)
    np.save('labels',labels)
    return data,labels

def flatten(data):
    return np.reshape(data,[-1,data.shape[1]*data.shape[2]]) #1d array

def PCAmethod(train_data,test_data,n):
    train_data=flatten(train_data)
    test_data=flatten(test_data)
    pca = PCA(n_components=n)
    return pca.fit_transform(train_data) , pca.transform(test_data)

def plot25(train_data):
    indexes = np.random.randint(0, train_data.shape[0], size=25)
    images = train_data[indexes]

    # plot 25 random digits
    plt.figure(figsize=(5, 5))
    for i in range(len(indexes)):
        plt.subplot(5, 5, i + 1)
        image = images[i]
        plt.imshow(image, cmap='gray')
        plt.axis('off')
def plot625(train_data):
    indexes = np.random.randint(0, train_data.shape[0], size=625)
    images = train_data[indexes]

    # plot 25 random digits
    plt.figure(figsize=(25, 25))
    for i in range(len(indexes)):
        plt.subplot(25, 25, i + 1)
        image = images[i]
        plt.imshow(image, cmap='gray')
        plt.axis('off')