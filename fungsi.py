import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import RandomRotation, RandomZoom, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

def make_model_mobile():
    img_size = 224
    batch_size_train = 32
    batch_size_test = 1
    
    inp = Input(shape = (224,224,3))
    model_mobile = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
    x1 = model_mobile(inp)
    x2 = GlobalAveragePooling2D()(x1)
    out = Dense(6, activation='softmax')(x2)
    
    model_mobile = Model(inputs = inp, outputs = out)
    model_mobile.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    model_mobile.fit(train_generator,
                     steps_per_epoch=STEP_SIZE_TRAIN,
                     epochs=20,
                     validation_data = test_generator
    )
    
    return mobile_model
