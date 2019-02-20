#####################################
### Imports
#####################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import json

import os

from datetime import datetime
from time import time
from argparse import ArgumentParser
from tensorflow import keras
from sklearn import metrics as metrics
from .datasets import TissueDataset, TissueDatasetFast


#####################################
### Consts
#####################################
MODEL_CHECKPOINT = './model_checkpoint.ckpt'
MODEL_FINAL = './model_final.hdf5'
GRAPH_ACC = './acc.png'
GRAPH_LOSS = './loss.png'
GRAPH_ROC = './roc.png'
LOG = './log.json'


def main():

    #####################################
    ### Program Args DEFAULT
    #####################################
    HDF5_FILE = './merged.hdf5'
    MODEL_USE = 1
    BATCH_SIZE_NEG=100
    BATCH_SIZE_POS=100
    BATCHES_PER_TRAIN_EPOCH = 50
    BATCHES_PER_VAL_EPOCH = 50
    EPOCHS = 25
    MAX_QUEUE_SIZE = 100
    L_RATE = 0.0001
    MASK_THRESHOLD = 0.1
    NORMALIZATION = 1
    WORKERS = None
    FAST_HDF5 = 0

    #####################################
    ### Arg Parser
    #####################################
    print('')
    print('########### Program Args')

    DESCRIPTION = 'Train CNN on merged HDF5'

    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        'hdf5', action='store', type=str, metavar='HDF5FILE',
        help='File path to hdf5 file'
    )

    parser.add_argument(
        '--arch', action='store', type=str, metavar='HDF5FILE',
        help='Architecture to use. 0=VGG16 (not working yet), 1=InceptionV3, 2=InceptionResNetV2.'
    )
    parser.add_argument(
        '--batch-size-neg', action='store', type=str, metavar='BATCHSIZENEG',
        help='Batch size for negative examples.'
    )
    parser.add_argument(
        '--batch-size-pos', action='store', type=str, metavar='BATCHSIZEPOS',
        help='Batch size for negative examples.'
    )
    parser.add_argument(
        '--batches-train', action='store', type=str, metavar='BATCHESTRAIN',
        help='Batches per train epoch.'
    )
    parser.add_argument(
        '--batches-val', action='store', type=str, metavar='BATCHESVAL',
        help='Batches per validation epoch.'
    )
    parser.add_argument(
        '--epochs', action='store', type=str, metavar='EPOCHS',
        help='Epochs to train.'
    )
    parser.add_argument(
        '--queue-size', action='store', type=str, metavar='QUEUESIZE',
        help='Number of batches to preload while training.'
    )
    parser.add_argument(
        '--l-rate', action='store', type=str, metavar='LRATE',
        help='Number of batches to preload while training.'
    )
    parser.add_argument(
        '--mask-threshold', action='store', type=str, metavar='MASKTHRESHOLD',
        help='Threshold of tissue in a cropped 224x224 tile to be valid. Or threshold of tumor tissue in a 224x224 slide to be valid.'
    )
    parser.add_argument(
        '--color-norm', action='store', type=str, metavar='COLORNORM',
        help='Colornormalization yes/no (1/0)'
    )
    parser.add_argument(
        '--workers', action='store', type=str, metavar='MULTIPROCESS',
        help='Number of parallel batch--generator workers. Setting this option enables multiprocessing.'
    )
    parser.add_argument(
        '--fast-hdf5', action='store', type=str, metavar='FASTHDF5',
        help='Which kind of HDF5 Format to you yes/no (1/0).'
    )

    args = parser.parse_args()

    HDF5_FILE = args.hdf5 if args.hdf5 else HDF5_FILE
    if not os.path.isfile(args.hdf5):
        HDF5_FILE = args.hdf5 + '/merged.hdf5'

    BATCH_SIZE_NEG = int(args.batch_size_neg) if args.batch_size_pos else BATCH_SIZE_NEG
    BATCH_SIZE_POS = int(args.batch_size_pos) if args.batch_size_pos else BATCH_SIZE_POS
    BATCHES_PER_TRAIN_EPOCH = int(args.batches_train) if args.batches_train else BATCHES_PER_TRAIN_EPOCH
    BATCHES_PER_VAL_EPOCH = int(args.batches_val) if args.batches_val else BATCHES_PER_VAL_EPOCH
    EPOCHS = int(args.epochs) if args.epochs else EPOCHS
    MAX_QUEUE_SIZE = int(args.queue_size) if args.queue_size else MAX_QUEUE_SIZE
    L_RATE = float(args.l_rate) if args.l_rate else L_RATE
    MASK_THRESHOLD = float(args.mask_threshold) if args.mask_threshold else MASK_THRESHOLD
    NORMALIZATION = float(args.color_norm) if args.color_norm else NORMALIZATION
    MODEL_USE = int(args.arch) if args.arch else MODEL_USE
    WORKERS = int(args.workers) if args.workers else WORKERS
    FAST_HDF5 = int(args.fast_hdf5) if args.fast_hdf5 else FAST_HDF5
    
    print('--hdf5', HDF5_FILE)
    print('--arch', MODEL_USE)
    print('--mask-threshold', MASK_THRESHOLD)
    print('--batch-size-neg', BATCH_SIZE_NEG)
    print('--batch-size-pos', BATCH_SIZE_POS)
    print('--batches-train', BATCHES_PER_TRAIN_EPOCH)
    print('--batches-val', BATCHES_PER_VAL_EPOCH)
    print('--epochs', EPOCHS)
    print('--queue-size', MAX_QUEUE_SIZE)
    print('--l-rate', L_RATE)
    print('--mask-threshold', MASK_THRESHOLD)
    print('--color-norm', NORMALIZATION)
    print('--workers', WORKERS)
    print('--fast-hdf5', FAST_HDF5)



    #####################################
    ### HDF5 - Loader and Batch Generator
    #####################################

    if FAST_HDF5:
        tsDsetToUse = TissueDatasetFast
    else:
        tsDsetToUse = TissueDataset

 





    #####################################
    ### Load Data Sample - Verbose
    #####################################
    print('')
    print('########### Getting one sample batch -verbose')

    train_data_tmp = tsDsetToUse(path=HDF5_FILE, validationset=False, verbose=True, threshold=MASK_THRESHOLD)
    val_data_tmp = tsDsetToUse(path=HDF5_FILE, validationset=True, verbose=True, threshold=MASK_THRESHOLD)


    itera = train_data_tmp.generator(num_neg=BATCH_SIZE_NEG, num_pos=BATCH_SIZE_POS, data_augm=True, normalize=NORMALIZATION)
    plt.figure(figsize=(12,4))
    for x, y in itera:
        print(x.shape)
        for i in range(2):
            ax = plt.subplot(1, 2, i + 1)
            plt.tight_layout()
            ax.set_title('Sample #{} - class {}'.format(i, y[i]))
            ax.imshow(x[i])
            ax.axis('off') 
        break # generate yields infinite random samples, so we stop after first





    #####################################
    ### Defining the Convolution Neural Network
    #####################################





    ### VGG-16 [DOES NOT WORK YET]
    if MODEL_USE == 0:
        base_model = keras.applications.VGG16(
                                         include_top=False, 
                                         weights='imagenet', 
                                         input_shape=(224,224,3), 
                                         )

        x = base_model.output
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(4096, activation='relu', name='fc1')(x)
        x = keras.layers.Dense(4096, activation='relu', name='fc2')(x)
        x = keras.layers.Dense(1, activation='sigmoid', name='predictions')(x)
        model = keras.Model(base_model, x, name='vgg16')

        #model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.0001), 
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])





    ### InceptionV3
    if MODEL_USE == 1:
        base_model = keras.applications.InceptionV3(
                                         include_top=False, 
                                         weights='imagenet', 
                                         input_shape=(224,224,3), 
                                         )

        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(1024, activation='relu')(x)
        predictions = keras.layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs=base_model.input, outputs=predictions)

        #model.compile(optimizer='adam',
        model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.0001), 
                      loss='binary_crossentropy',
                      metrics=['accuracy'])





    ### InceptionResNetV2
    if MODEL_USE == 2:
        base_model = keras.applications.InceptionResNetV2(
                                         include_top=False, 
                                         weights='imagenet', 
                                         input_shape=(224,224,3), 
                                         )

        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(1024, activation='relu')(x)
        predictions = keras.layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs=base_model.input, outputs=predictions)

        #model.compile(optimizer='adam',
        model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=L_RATE), 
                      loss='binary_crossentropy',
                      metrics=['accuracy'])





    #####################################
    ### If you interrupted your training load the checkpoint here, e.g.:
    #####################################
    #model.load_weights(MODEL_CHECKPOINT)





    #####################################
    ### Create callbacks
    #####################################
    cp_callback = tf.keras.callbacks.ModelCheckpoint(MODEL_CHECKPOINT, 
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     verbose=1)

    class TimeHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []
            self.starts = []
            self.ends = []

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time()

        def on_epoch_end(self, batch, logs={}):
            now = time()
            self.times.append(now - self.epoch_time_start)
            self.starts.append(self.epoch_time_start)
            self.ends.append(now)
            
    time_callback = TimeHistory()



    #####################################
    ### Training / Validation
    #####################################
    print('')
    print('########### Start real training')



    train_data = tsDsetToUse(path=HDF5_FILE, validationset=False, verbose=False, threshold=MASK_THRESHOLD)
    val_data = tsDsetToUse(path=HDF5_FILE, validationset=True, verbose=False, threshold=MASK_THRESHOLD)

    use_multiprocessing = None
    workers = 1
    if WORKERS is not None:
        workers = WORKERS
        use_multiprocessing = True

    hist = model.fit_generator(
            generator=train_data.generator(BATCH_SIZE_NEG, BATCH_SIZE_POS, True, NORMALIZATION),
            steps_per_epoch=BATCHES_PER_TRAIN_EPOCH, 
            validation_data=val_data.generator(BATCH_SIZE_NEG, BATCH_SIZE_POS, False, NORMALIZATION),
            validation_steps=BATCHES_PER_VAL_EPOCH,
            epochs=EPOCHS,
            callbacks=[time_callback, cp_callback], 
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=MAX_QUEUE_SIZE)

    #####################################
    ### Print Summary
    #####################################
    print('')
    print('########### Print Train summary')
    print(hist.history)






    #####################################
    ### Save entire model to a HDF5 file
    #####################################
    model.save(MODEL_FINAL)




    #####################################
    ### FINAL VALIDATION
    #####################################
    print('')
    print('########### Start eval final for conf_matrix')

    y_train = []
    preds_train = []
    y_val = []
    preds_val = []

    train_data_final_val = tsDsetToUse(path=HDF5_FILE, validationset=False, verbose=False, threshold=MASK_THRESHOLD)
    val_data_final_val = tsDsetToUse(path=HDF5_FILE, validationset=True, verbose=False, threshold=MASK_THRESHOLD)

    for i in range(BATCHES_PER_TRAIN_EPOCH + BATCHES_PER_VAL_EPOCH):
        x_train, y_train_tmp = train_data_final_val.get_batch(BATCH_SIZE_NEG, BATCH_SIZE_POS, True, NORMALIZATION)
        preds_train += (model.predict(x_train, batch_size=(BATCH_SIZE_NEG + BATCH_SIZE_POS), verbose=1).tolist())
        y_train += (y_train_tmp.tolist())

        x_val, y_val_tmp = val_data_final_val.get_batch(BATCH_SIZE_NEG, BATCH_SIZE_POS, False, NORMALIZATION)
        preds_val += (model.predict(x_val, batch_size=(BATCH_SIZE_NEG + BATCH_SIZE_POS), verbose=1).tolist())
        y_val += (y_val_tmp.tolist())

    fpr_train, tpr_train, thresholds_trian = metrics.roc_curve(y_train, preds_train)
    fpr_val, tpr_val, thresholds_val = metrics.roc_curve(y_val, preds_val)
    auc_train = metrics.roc_auc_score(y_train, preds_train)
    auc_val = metrics.roc_auc_score(y_val, preds_val)

    preds_train = np.array(preds_train)
    preds_val = np.array(preds_val)
    preds_train[preds_train >= 0.5] = 1
    preds_train[preds_train < 0.5] = 0
    preds_val[preds_val >= 0.5] = 1
    preds_val[preds_val < 0.5] = 0

    f1_train = metrics.f1_score(y_train, preds_train)
    f1_val = metrics.f1_score(y_val, preds_val)

    cm_train = metrics.confusion_matrix(y_train, preds_train)
    cm_val = metrics.confusion_matrix(y_val, preds_val)

    TP_train = cm_train[1,1]
    TN_train = cm_train[0,0]
    FP_train = cm_train[0,1]
    FN_train = cm_train[1,0]
    TP_val = cm_val[1,1]
    TN_val = cm_val[0,0]
    FP_val = cm_val[0,1]
    FN_val = cm_val[1,0]
    final_acc_train = (TP_train + TN_train) / (FN_train + FP_train + TP_train + TN_train)
    final_acc_val = (TP_val + TN_val) / (FN_val + FP_val + TP_val + TN_val)

    #####################################
    ### Logging
    #####################################
    print('')
    print('########### Generating Log')

    CSV_HEADER = [
        'mask_threshold', 
        'batch_size_pos', 
        'batch_size_neg', 
        'batches_per_train_epoch',  
        'batches_per_val_epoch', 
        'epochs', 
        'model_architecture',
        'queue_size',
        'l_rate',
        'color_corm',
        'workers',
        'fast_hdf5'
        ]

    json_log = {
        'program_args': {}
    }

    second_line_tmp = [MASK_THRESHOLD, BATCH_SIZE_POS, BATCH_SIZE_NEG, BATCHES_PER_TRAIN_EPOCH, BATCHES_PER_VAL_EPOCH, EPOCHS, MODEL_USE, MAX_QUEUE_SIZE, L_RATE, NORMALIZATION, WORKERS, FAST_HDF5]
    for i in range(len(second_line_tmp)):
        json_log['program_args'][CSV_HEADER[i]] = second_line_tmp[i]
        
    json_log['train_summary'] = hist.history
    json_log['train_summary']['train_batch_times'] = train_data.batch_times
    json_log['train_summary']['val_batch_times'] = val_data.batch_times
    json_log['train_summary']['epochs_durations'] = time_callback.times
    json_log['train_summary']['epochs_starts'] = time_callback.starts
    json_log['train_summary']['epochs_ends'] = time_callback.ends

    json_log['final_model'] = {}
    json_log['final_model']['tp_train'] = TP_train
    json_log['final_model']['tn_train'] = TN_train
    json_log['final_model']['fp_train'] = FP_train
    json_log['final_model']['fn_train'] = FN_train
    json_log['final_model']['tp_val'] = TP_val
    json_log['final_model']['tn_val'] = TN_val
    json_log['final_model']['fp_val'] = FP_val
    json_log['final_model']['fn_val'] = FN_val
    json_log['final_model']['f1_train'] = f1_train
    json_log['final_model']['f1_val'] = f1_val
    json_log['final_model']['acc_train'] = final_acc_train
    json_log['final_model']['acc_val'] = final_acc_val
    json_log['final_model']['fpr_train'] = fpr_train.tolist()
    json_log['final_model']['fpr_val'] = fpr_val.tolist()
    json_log['final_model']['tpr_train'] = tpr_train.tolist()
    json_log['final_model']['tpr_val'] = tpr_val.tolist()
    json_log['final_model']['threshold_train'] = thresholds_trian.tolist()
    json_log['final_model']['threshold_val'] = thresholds_val.tolist()
    json_log['final_model']['auc_train'] = auc_train
    json_log['final_model']['auc_val'] = auc_val

    def default(o):
        if isinstance(o, np.int64): return int(o)  
        raise TypeError

    with open(LOG, 'w') as fp:
        json.dump(json_log, fp, default=default)
        
        
        
        
    #####################################
    ### Plots
    #####################################
    print('')
    print('########### Generating Plots')


    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.plot(np.linspace(0, len(hist.history['acc']), len(hist.history['acc'])), 
            hist.history['acc'], 
            label='train')
    plt.plot(np.linspace(0, len(hist.history['val_acc']), len(hist.history['val_acc'])), 
            hist.history['val_acc'], 
            label='valid')
    plt.legend(loc=2)
    plt.savefig(GRAPH_ACC)
    plt.show()
    plt.close()


    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(np.linspace(0, len(hist.history['loss']), len(hist.history['loss'])), 
            hist.history['loss'],
            label='train')
    plt.plot(np.linspace(0, len(hist.history['val_loss']), len(hist.history['val_loss'])), 
            hist.history['val_loss'], 
            label='valid')
    plt.legend(loc=2)
    plt.savefig(GRAPH_LOSS)
    plt.show()
    plt.close()

    plt.ylabel('tpr')
    plt.xlabel('fpr')
    plt.title('roc')
    plt.plot(
            fpr_train,
            tpr_train,
            label='train')
    plt.plot(
            fpr_val, 
            tpr_val, 
            label='valid')


    plt.legend(loc=2)
    plt.savefig(GRAPH_ROC)
    plt.show()
    plt.close()
