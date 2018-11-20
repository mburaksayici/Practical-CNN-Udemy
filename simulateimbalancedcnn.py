from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator



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






model.compile(optimizer=Adam(lr=1e-4),loss="categorical_crossentropy",metrics=['accuracy'])

batch_size = 50
epochs =7







model.fit_generator(train_flow,
                    steps_per_epoch=len(train_flow) / batch_size,
                    validation_data=val_flow,
                    validation_steps=len(val_flow) / batch_size,
                    epochs=epochs, class_weight={0: 0.3,1:0.7},
                    # callbacks=[reduce, tb, early],
                    verbose=1
                    )

