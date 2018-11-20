import os
import random
import numpy as np
import cv2
from imgaug import augmenters as iaa





seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                ]),
    iaa.Flipud(0.2),# horizontally flip
    iaa.SomeOf(2, [
        iaa.Affine(rotate=(-40, 40), translate_percent={"x": (-0.25, 0.25)}, mode='symmetric', cval=(0)),
        iaa.PerspectiveTransform(scale=(0.06, 0.12)),
        iaa.PiecewiseAffine(scale=(0.02, 0.04), mode='edge', cval=(0)),
    ]),

])
seq_det = seq.to_deterministic()



def do_augmentation(sequent,image):
    augimage = np.array(sequent.augment_image(image))
    return np.reshape(augimage,(-1,299,299,3))











def getimagefrompath(path,imgbyclass):
    classnumb = len(os.listdir(path))
    inputarray = np.array([])
    outputarray = np.array([])
    outcontrol = 0
    for element in os.listdir(path):
        classpath = path + element+"/"
        count = 0
        while count<imgbyclass :
            random_filename = random.choice([x for x in os.listdir(classpath) if os.path.isfile(os.path.join(classpath, x))])
            imagepath = classpath+random_filename
            image = cv2.imread(imagepath)
            image = cv2.resize(image,(299,299))/255
            inputarray = np.append(inputarray,image)
            output = np.zeros((1,classnumb))

            output[0,outcontrol] = 1
            print(output, " --- ", imagepath)
            outputarray = np.append(outputarray,output)
            count +=1
        outcontrol +=1
    return inputarray.reshape(-1,299,299,3),outputarray.reshape(-1,classnumb)





def generator(path,imgbyclass,sequent):
    while True:
        inputdata,outputdata = getimagefrompath(path,imgbyclass)

        inputdata = np.reshape(inputdata,(-1,299,299,3))
        for i in range(np.shape(outputdata)[0]):
            inputdata[i]= do_augmentation(sequent,inputdata[i])
        yield (inputdata,outputdata)
        


path = "kerasimagedatagen/Train/"
generator("kerasimagedatagen/Train/",3,seq_det)
