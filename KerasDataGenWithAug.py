from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import TakeRandomImages
from TakeRandomImages import *


base_model = InceptionV3(weights="imagenet",include_top=False)


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512,activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512,activation="relu")(x)
predictions = Dense(2,activation="softmax")(x)


model = Model(inputs = base_model.input,outputs = predictions)



for layer in model.layers[:309]:
    layer.trainable = False
for layer in model.layers[309:]:
    layer.trainable= True






model.compile(optimizer=Adam(lr=1e-5),loss="categorical_crossentropy",metrics=['accuracy'])

batch_size = 10
epochs =10





train_datagen = ImageDataGenerator(zca_whitening=True,width_shift_range=0.1,horizontal_flip=True,vertical_flip=True,rotation_range=180,rescale=1/255)
val_datagen = ImageDataGenerator(rescale=1/255)

train_flow = train_datagen.flow_from_directory(directory="kerasimagedatagen/Train/",
    target_size=(299, 299),shuffle = True,
    batch_size=batch_size,
    class_mode="categorical",)

val_flow = val_datagen.flow_from_directory(directory="kerasimagedatagen/Val/",
    target_size=(299, 299),
    batch_size=batch_size,

    class_mode="categorical")




model.fit_generator(generator=train_flow,
                    steps_per_epoch=25000 / batch_size,
                    validation_data=val_flow,
                    validation_steps=60 / batch_size,
                    epochs=10,
                    # callbacks=[reduce, tb, early],
                    verbose=1
                    )


