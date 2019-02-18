#####################################
### Imports
#####################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import h5py
import json

import os

from datetime import datetime
from time import time
from argparse import ArgumentParser
from tensorflow import keras
from sklearn import metrics as metrics


#####################################
### Consts
#####################################
MODEL_CHECKPOINT = './model_checkpoint.ckpt'
MODEL_FINAL = './model_final.hdf5'
GRAPH_ACC = './acc.png'
GRAPH_LOSS = './loss.png'
GRAPH_ROC = './roc.png'
LOG  = './log.json'


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
    USE_MULTI_PROCESS = 0



    #'''
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
        '--multi-process', action='store', type=str, metavar='MULTIPROCESS',
        help='Parallel batch--generator yes/no (1/0).'
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
    USE_MULTI_PROCESS = int(args.multi_process) if args.arch else USE_MULTI_PROCESS
    #'''
    
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
    print('--l-color-norm', NORMALIZATION)
    print('--l-multi-process', USE_MULTI_PROCESS)



    #####################################
    ### HDF5 - Loader and Batch Generator
    #####################################
    class TissueDataset():
        """Data set for preprocessed WSIs of the CAMELYON16 and CAMELYON17 data set."""

        def __init__(self, path, validationset=False, verbose=False):      
            self.h5 = h5py.File(path, 'r', libver='latest', swmr=True)
            self.tilesize = 224
            self.verbose = verbose
            self.normals = []
            self.tumors = []
            for group in self.h5:
                for sub in self.h5[group]:
                    if 'umor' in sub:
                        self.tumors.append('tumor/' + sub)
                    elif 'ormal' in sub:
                        self.normals.append('normal/' + sub)
                    for subsub in self.h5[group][sub]:
                        dset = self.h5[group][sub][subsub]
            if validationset:
                if verbose: print('validation set:')
                self.normals = self.normals[1::2]
                self.tumors = self.tumors[1::2]
            else:
                if verbose: print('training set:')
                self.normals = self.normals[::2]
                self.tumors = self.tumors[::2]
            self.n_tumors = len(self.tumors)
            self.n_normals = len(self.normals)
            if self.verbose: print(self.normals)
            if self.verbose: print(self.tumors)
            self.get_batch_call_start = []
            self.get_batch_call_end = []
            self.get_batch_call_diff = []
            
        def get_batch_call_times(self):
            return self.get_batch_call_start, self.get_batch_call_end,  self.get_batch_call_diff

        def __get_tiles_from_path(self, dataset_names, max_wsis, number_tiles):
            tiles = np.ndarray((number_tiles, self.tilesize, self.tilesize, 3))
            for i in range(number_tiles):
                if self.verbose: print('gettin_tile_nr ', i)
                valid_tile = False
                while valid_tile == False:            
                    wsi_idx = np.random.randint(0, max_wsis)
                    wsi = self.h5[dataset_names[wsi_idx]]
                    zoom_lvl = 0
                    n_tiles = len(wsi['img'][zoom_lvl, :, 0, 0])
                    if n_tiles >= 1:
                        #n_tiles = len(wsi['img'][zoom_lvl]) ##### SUPERSLOW
                        tile_idx = np.random.randint(0, n_tiles)
                        mask = wsi['mask'][zoom_lvl, tile_idx]
                        #mask = wsi['mask'][zoom_lvl][tile_idx] ##### SPERSLOW
                        ### crop random 256x256
                        hdf5_tilesize = mask.shape[0]
                        trys_to_get_valid_tile = 0
                        while trys_to_get_valid_tile < 10:
                            if self.verbose: print('trys_to_get_valid_tile ', trys_to_get_valid_tile)
                            rand_height = np.random.randint(0, hdf5_tilesize-self.tilesize)
                            rand_width = np.random.randint(0, hdf5_tilesize-self.tilesize)
                            mask_cropped = mask[rand_height:rand_height+self.tilesize, rand_width:rand_width+self.tilesize]
                            if mask_cropped.sum() > (self.tilesize*self.tilesize * MASK_THRESHOLD):
                                tile_cropped = wsi['img'][zoom_lvl, tile_idx, rand_height:rand_height+self.tilesize,rand_width:rand_width+self.tilesize]
                                tiles[i] = tile_cropped
                                trys_to_get_valid_tile = 1337
                                valid_tile = True
                            trys_to_get_valid_tile += 1

            tiles = tiles / 255.
            return tiles

        def __get_random_positive_tiles(self, number_tiles):
            if self.verbose: print('getting_tumor_tile')
            return self.__get_tiles_from_path(self.tumors, self.n_tumors, number_tiles), np.ones((number_tiles))

        def __get_random_negative_tiles(self, number_tiles):
            if self.verbose: print('getting_normal_tile')
            return self.__get_tiles_from_path(self.normals, self.n_normals, number_tiles), np.zeros((number_tiles))

        def generator(self, num_neg=10, num_pos=10, data_augm=False, normalize=False):
            while True:
                x, y = self.get_batch(num_neg, num_pos, data_augm, normalize)
                yield x, y

        def get_batch(self, num_neg=10, num_pos=10, data_augm=False, normalize=False):
            now1 = time()
            self.get_batch_call_start.append(now1)
            x_p, y_p = self.__get_random_positive_tiles(num_pos)
            x_n, y_n = self.__get_random_negative_tiles(num_neg)
            x = np.concatenate((x_p, x_n), axis=0)
            y = np.concatenate((y_p, y_n), axis=0)

            if data_augm:
                if np.random.randint(0,2): x = np.flip(x, axis=1)
                if np.random.randint(0,2): x = np.flip(x, axis=2)
                x = np.rot90(m=x, k=np.random.randint(0,4), axes=(1,2))
            if normalize:
                x[:,:,:,0] = (x[:,:,:,0] - np.mean(x[:,:,:,0])) / np.std(x[:,:,:,0])
                x[:,:,:,1] = (x[:,:,:,1] - np.mean(x[:,:,:,1])) / np.std(x[:,:,:,1])
                x[:,:,:,2] = (x[:,:,:,2] - np.mean(x[:,:,:,2])) / np.std(x[:,:,:,2])

            p = np.random.permutation(len(y))
            now2 = time()
            self.get_batch_call_end.append(now2)
            self.get_batch_call_diff.append(now2-now1)
            if self.verbose: print('batch_generator (start, end, diff): ',now1,now2,now2-now1)
            if self.verbose: print('batch: ', x.shape)
            return x[p], y[p]





    #####################################
    ### Load Data Sample - Verbose
    #####################################
    print('')
    print('########### Getting one sample batch -verbose')

    train_data_tmp = TissueDataset(path=HDF5_FILE, validationset=False, verbose=True)
    val_data_tmp = TissueDataset(path=HDF5_FILE, validationset=True, verbose=True)


    itera = train_data_tmp.generator(num_neg=BATCH_SIZE_NEG, num_pos=BATCH_SIZE_POS, data_augm=True, normalize=NORMALIZATION)
    #plt.figure(figsize=(12,4))
    #for x, y in itera:
    #    print(x.shape)
    #    for i in range(2):
    #        ax = plt.subplot(1, 2, i + 1)
    #        plt.tight_layout()
    #        ax.set_title('Sample #{} - class {}'.format(i, y[i]))
    #        ax.imshow(x[i])
    #        ax.axis('off') 
    #    break # generate yields infinite random samples, so we stop after first





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

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time()

        def on_epoch_end(self, batch, logs={}):
            self.times.append(time() - self.epoch_time_start)
            
    time_callback = TimeHistory()



    #####################################
    ### Training / Validation
    #####################################
    print('')
    print('########### Start real training')



    train_data = TissueDataset(path=HDF5_FILE, validationset=False, verbose=False)
    val_data = TissueDataset(path=HDF5_FILE, validationset=True, verbose=False)

    now1 = datetime.now()
    hist = model.fit_generator(
            generator=train_data.generator(BATCH_SIZE_NEG, BATCH_SIZE_POS, True, NORMALIZATION),
            steps_per_epoch=BATCHES_PER_TRAIN_EPOCH, 
            validation_data=val_data.generator(BATCH_SIZE_NEG, BATCH_SIZE_POS, False, NORMALIZATION),
            validation_steps=BATCHES_PER_VAL_EPOCH,
            epochs=EPOCHS,
            callbacks=[time_callback, cp_callback], 
            workers=6, 
            use_multiprocessing=USE_MULTI_PROCESS, 
            max_queue_size=MAX_QUEUE_SIZE)
    now2 = datetime.now()


    total_training_time = now2-now1
    training_get_batch_calls = train_data.get_batch_call_times()
    validation_get_batch_calls = val_data.get_batch_call_times()




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
    for i in range(BATCHES_PER_TRAIN_EPOCH + BATCHES_PER_VAL_EPOCH):
        x_train, y_train_tmp = train_data.get_batch(BATCH_SIZE_NEG, BATCH_SIZE_POS, True, NORMALIZATION)
        preds_train += (model.predict(x_train, batch_size=(BATCH_SIZE_NEG + BATCH_SIZE_POS), verbose=1).tolist())
        y_train += (y_train_tmp.tolist())

        x_val, y_val_tmp = val_data.get_batch(BATCH_SIZE_NEG, BATCH_SIZE_POS, False, NORMALIZATION)
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
    json_log = {}
    json_log['program_args'] = {}

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
        'multi_process'
        ] 

    second_line_tmp = [MASK_THRESHOLD, BATCH_SIZE_POS, BATCH_SIZE_NEG, BATCHES_PER_TRAIN_EPOCH, BATCHES_PER_VAL_EPOCH, EPOCHS, MODEL_USE, MAX_QUEUE_SIZE, L_RATE, NORMALIZATION, USE_MULTI_PROCESS]
    for i in range(len(second_line_tmp)):
        json_log['program_args'][CSV_HEADER[i]] = second_line_tmp[i]
        
    json_log['train_summary'] = hist.history
    json_log['train_summary']['train_get_batch_calls_start'] = str(training_get_batch_calls[0])
    json_log['train_summary']['train_get_batch_calls_end'] = str(training_get_batch_calls[1])
    json_log['train_summary']['train_get_batch_calls_diff'] = str(training_get_batch_calls[2])
    json_log['train_summary']['val_get_batch_calls_start'] = str(validation_get_batch_calls[0])
    json_log['train_summary']['val_get_batch_calls_end'] = str(validation_get_batch_calls[1])
    json_log['train_summary']['val_get_batch_calls_diff'] = str(validation_get_batch_calls[2])
    json_log['train_summary']['epochs_durations'] = time_callback.times

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

    json_log = json.dumps(json_log, default=default)
    with open(LOG, 'w') as fp:
        json.dump(json_log, fp)
        
        
        
        
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
