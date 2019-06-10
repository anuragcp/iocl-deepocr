from keras import backend as K
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Image_Generator import TextImageGenerator
from Model import get_Model
from parameter import *
K.set_learning_phase(0)

# # Model description and training

model = get_Model(training=True)

try:
    model.load_weights('LSTM+BN4--26--0.011.hdf5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass

train_file_path = './DB/train/' 
tiger_train = TextImageGenerator(train_file_path, img_w, img_h, batch_size,downsample_factor,600)   # train only first 50000 images
tiger_train.build_data('train/data.csv')

valid_file_path = './DB/test/'
tiger_val = TextImageGenerator(valid_file_path, img_w, img_h, val_batch_size,downsample_factor,50)    # give 45 images for validation
tiger_val.build_data('test/data.csv')

ada = Adadelta()

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
checkpoint = ModelCheckpoint(filepath='LSTM+BN5--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='min', period=1) #chechkpoint
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

print(tiger_val.n / val_batch_size)
# captures output of softmax so we can decode the output during visualization
model.fit_generator(generator=tiger_train.next_batch(),
                    steps_per_epoch=int(tiger_train.n / batch_size),
                    epochs=5, # 5 epochs
                    callbacks=[checkpoint],
                    validation_data=tiger_val.next_batch(),
                    validation_steps=int(tiger_val.n / val_batch_size))
