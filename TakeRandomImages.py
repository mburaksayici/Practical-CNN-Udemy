
import numpy as np
import cv2
import random
import os
from imgaug import augmenters as iaa
path = "kerasimagedatagen/Train/"

#print(os.listdir(path))



seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontally flip
    iaa.SomeOf(3,[
        iaa.GaussianBlur(sigma=(0.0, 1.0)),
        iaa.Affine(rotate=(-30, 30), translate_percent={"x": (-0.25, 0.25)}, mode='symmetric', cval=(0)),
        iaa.PerspectiveTransform(scale=(0.04, 0.08)),
        iaa.PiecewiseAffine(scale=(0.05, 0.1), mode='edge', cval=(0)),
    ]),

])
seq_det = seq.to_deterministic()


def do_augmentation(seq_det, image):
    image = np.array(seq_det.augment_image(image))
    return image.reshape(-1,299,299,3)





def getimagefrompath(path,imgbyclass):
    path = "kerasimagedatagen/Train/"
    classnumb = len(os.listdir(path))
    inputarray = np.array([])
    outputarray = np.array([])
    outcontrol = 0
    for elements in os.listdir(path):
        classpath = path+ elements + "/"
        count=0


        while count < imgbyclass:

            random_filename = random.choice([x for x in os.listdir(classpath) if os.path.isfile(os.path.join(classpath, x))])

            image = cv2.imread(classpath+ str(random_filename))
            image = cv2.resize(image,(299,299))/255 #Normalize

            inputarray = np.append(inputarray,image)
            output = np.zeros((1, classnumb))

            output[0,outcontrol] = 1

            outputarray = np.append(outputarray,output)


            #print(image)
            count += 1
            #print(classpath+ str(random_filename), "corresponds to this output", output)
            #cv2.imshow("deneme",image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        outcontrol +=1
    return inputarray,outputarray.reshape(-1,classnumb)











def generator(path,imgbyclass):
    while True:
        inputdata, outputdata = getimagefrompath(path,imgbyclass)

        inputdata = np.reshape(inputdata,(-1,299,299,3))
        for i in range(np.shape(outputdata)[0]):

            #cv2.imshow("ew", inputdata[i])
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            inputdata[i] = do_augmentation(seq_det,inputdata[i])


            #cv2.imshow("ew", inputdata[i])
            #cv2.waitKey(0)
            # cv2.destroyAllWindows()

        yield (inputdata,outputdata)


