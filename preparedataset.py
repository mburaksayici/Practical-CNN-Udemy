"""
1- Firstly find CNN
2- Identify the image size
3- To load the dataset, prepare the data:
    Take the adress of images, then load it with opencv
4- Load images with  OPENCV/SCİPY
5- Resize 224,224
6- Save it to numpy format
7- We need to prepare output()
"""


import glob
from scipy import misc
import numpy as np


catimagenumber = 0


catdata = np.array([])




for imagepath in glob.glob("kaggledata/PetImages/cat2/*"):
    try:
        image = misc.imread(imagepath)
        resizedimage = misc.imresize(image,(224,224))
        catdata = np.append(catdata,resizedimage)

        catimagenumber = catimagenumber +1
    except:
        print(imagepath)



print(int(catdata.shape[0])/(224*224*3))
print(catimagenumber)


print(catimagenumber)
np.save("catdata",catdata)


#####--------

dogimagenumber = 0


dogdata = np.array([])

for imagepath in glob.glob("kaggledata/PetImages/dog2/*"):
    try:
        image = misc.imread(imagepath)
        resizedimage = misc.imresize(image,(224,224))
        dogdata = np.append(dogdata,resizedimage)

        dogimagenumber = dogimagenumber +1
    except:
        print(imagepath)



print(int(dogdata.shape[0])/(224*224*3))
print(dogimagenumber)
np.save("dogdata",dogdata)


catlabel = np.array([1,0])
catoutput =  np.tile(catlabel,int(catdata.shape[0]/(224*224*3)))
print(catoutput)

doglabel = np.array([0,1])
dogoutput =  np.tile(doglabel,int(dogdata.shape[0]/(224*224*3)))

print(dogoutput)