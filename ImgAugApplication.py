from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import takerandimages
from takerandimages import generator, seq_det



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
epochs=10

classnumb = 3
batch_size = 2*classnumb

model.fit_generator(generator=generator("kerasimagedatagen/Train/",5,sequent=seq_det),
                    steps_per_epoch=25000 / batch_size,
                    validation_data=generator("kerasimagedatagen/Train/",5,sequent=seq_det),
                    validation_steps=60 / batch_size,
                    epochs=10,
                    # callbacks=[reduce, tb, early],
                    verbose=1
                    )