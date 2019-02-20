import h5py
import numpy as np
from uuid import uuid4
from time import time

class TissueDataset():
    """Data set for preprocessed WSIs of the CAMELYON16 and CAMELYON17 data set."""

    def __init__(self, path, validationset=False, verbose=False, threshold=0.1):      
        self.h5 = h5py.File(path, 'r', libver='latest', swmr=True)
        self.tilesize = 224
        self.verbose = verbose
        self.normals = []
        self.tumors = []
        self.threshold = threshold
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
        self.batch_times = {}
        
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
                    tile_idx = np.random.randint(0, n_tiles)
                    mask = wsi['mask'][zoom_lvl, tile_idx]
                    ### crop random 256x256
                    hdf5_tilesize = mask.shape[0]
                    trys_to_get_valid_tile = 0
                    while trys_to_get_valid_tile < 10:
                        if self.verbose: print('trys_to_get_valid_tile ', trys_to_get_valid_tile)
                        rand_height = np.random.randint(0, hdf5_tilesize-self.tilesize)
                        rand_width = np.random.randint(0, hdf5_tilesize-self.tilesize)
                        mask_cropped = mask[rand_height:rand_height+self.tilesize, rand_width:rand_width+self.tilesize]
                        if mask_cropped.sum() > (self.tilesize*self.tilesize * self.threshold):
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
        gen_id = str(uuid4())
        self.batch_times[gen_id] = []
        while True:
            start_time = time()
            x, y = self.get_batch(num_neg, num_pos, data_augm, normalize)
            end_time = time()
            self.batch_times[gen_id].append((start_time, end_time))
            yield x, y

    def get_batch(self, num_neg=10, num_pos=10, data_augm=False, normalize=False):
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
        return x[p], y[p]


class TissueDatasetFast():
    """Data set for preprocessed WSIs of the CAMELYON16 and CAMELYON17 data set."""

    def __init__(self, path, validationset=False, verbose=False, threshold=0.1):      
        self.h5 = h5py.File(path, 'r', libver='latest', swmr=True)
        self.tilesize = 224
        self.verbose = verbose
        self.threshold = threshold
        if validationset:
            if verbose: print('validation set:')
            self.tumors = self.h5['tumor_train']
            self.normals = self.h5['normal_train']
        else:
            if verbose: print('training set:')
            self.tumors = self.h5['tumor_valid']
            self.normals = self.h5['normal_valid']
        self.n_tumors = self.tumors.shape[0]
        self.n_normals = self.normals.shape[0]
        self.batch_times = {}
        
    def __get_tiles_from_path(self, dataset, max_wsis, number_tiles):
        hdf5_tilesize = dataset.shape[1]
        wsi_idx = np.random.randint(0, max_wsis-number_tiles)
        rand_height = np.random.randint(0, hdf5_tilesize-self.tilesize)
        rand_width = np.random.randint(0, hdf5_tilesize-self.tilesize)
        tiles = dataset[wsi_idx:wsi_idx+number_tiles, rand_height:rand_height+self.tilesize, rand_width:rand_width+self.tilesize]
        tiles = tiles / 255.
        return tiles

    def __get_random_positive_tiles(self, number_tiles):
        if self.verbose: print('getting_tumor_tile')
        return self.__get_tiles_from_path(self.tumors, self.n_tumors, number_tiles), np.ones((number_tiles))

    def __get_random_negative_tiles(self, number_tiles):
        if self.verbose: print('getting_normal_tile')
        return self.__get_tiles_from_path(self.normals, self.n_normals, number_tiles), np.zeros((number_tiles))

    def generator(self, num_neg=10, num_pos=10, data_augm=False, normalize=False):
        gen_id = str(uuid4())
        self.batch_times[gen_id] = []
        while True:
            start_time = time()
            x, y = self.get_batch(num_neg, num_pos, data_augm, normalize)
            end_time = time()
            self.batch_times[gen_id].append((start_time, end_time))
            yield x, y

    def get_batch(self, num_neg=10, num_pos=10, data_augm=False, normalize=False):
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
        return x[p], y[p]


class TissueDatasetFastWithValidation():
    """Data set for preprocessed WSIs of the CAMELYON16 and CAMELYON17 data set."""

    def __init__(self, path, validationset=False, verbose=False, threshold=0.1):      
        self.h5 = h5py.File(path, 'r', libver='latest', swmr=True)
        self.tilesize = 224
        self.verbose = verbose
        self.threshold = threshold
        if validationset:
            if verbose: print('validation set:')
            self.tumors = self.h5['tumor_train']
            self.normals = self.h5['normal_train']
        else:
            if verbose: print('training set:')
            self.tumors = self.h5['tumor_valid']
            self.normals = self.h5['normal_valid']
        self.n_tumors = self.tumors.shape[0]
        self.n_normals = self.normals.shape[0]
        self.batch_times = {}
        
    def __get_tiles_from_path(self, dataset, max_wsis, number_tiles):
        hdf5_tilesize = dataset.shape[1]
        wsi_idx = np.random.randint(0, max_wsis-number_tiles)
        rand_height = np.random.randint(0, hdf5_tilesize-self.tilesize)
        rand_width = np.random.randint(0, hdf5_tilesize-self.tilesize)

        #masks = dataset[wsi_idx:wsi_idx+number_tiles, rand_height:rand_height+self.tilesize, rand_width:rand_width+self.tilesize,-1]
        #tiles = dataset[wsi_idx:wsi_idx+number_tiles, rand_height:rand_height+self.tilesize, rand_width:rand_width+self.tilesize,0:-1]

        loaded = dataset[wsi_idx:wsi_idx+number_tiles, rand_height:rand_height+self.tilesize, rand_width:rand_width+self.tilesize,:]
        masks = loaded[:, :, :, -1]
        tiles = loaded[:, :, :, :-1]

        valid_idxs = []
        for i in range(masks.shape[0]):
            if masks[i].sum() > self.threshold * (self.tilesize**2):
                valid_idxs.append(i)
        tiles = tiles[valid_idxs]
        tiles = tiles / 255.
        return tiles

    def __get_random_positive_tiles(self, number_tiles):
        if self.verbose: print('getting_tumor_tile')
        return self.__get_tiles_from_path(self.tumors, self.n_tumors, number_tiles)

    def __get_random_negative_tiles(self, number_tiles):
        if self.verbose: print('getting_normal_tile')
        return self.__get_tiles_from_path(self.normals, self.n_normals, number_tiles)

    def generator(self, num_neg=10, num_pos=10, data_augm=False, normalize=False, data_slice_size=100):
        gen_id = str(uuid4())
        self.batch_times[gen_id] = []
        caches = {
            'pos_cache': None,
            'neg_cache': None
        }
        while True:
            start_time = time()
            x, y = self.get_batch(
                caches,
                num_pos=num_pos,
                num_neg=num_neg,
                data_augm=data_augm,
                normalize=normalize,
                data_slice_size=data_slice_size
            )
            end_time = time()
            self.batch_times[gen_id].append((start_time, end_time))
            yield x, y

    def get_batch(self, caches, num_neg=10, num_pos=10, data_augm=False, normalize=False, data_slice_size=100):
        # fill positive tiles cache
        while True:
            if caches['pos_cache'] is None:
                caches['pos_cache'] = self.__get_random_positive_tiles(data_slice_size)

            if caches['pos_cache'].shape[0] >= num_pos:
                break

            np.concatenate((caches['pos_cache'], self.__get_random_positive_tiles(data_slice_size)), axis=0)

        # fill negative tiles cache
        while True:
            if caches['neg_cache'] is None:
                caches['neg_cache'] = self.__get_random_negative_tiles(data_slice_size)

            if caches['neg_cache'].shape[0] >= num_neg:
                break

            np.concatenate((caches['neg_cache'], self.__get_random_negative_tiles(data_slice_size)), axis=0)

        # take tiles from cache
        x_p = caches['pos_cache'][:num_pos]
        x_n = caches['neg_cache'][:num_neg]

        caches['pos_cache'] = caches['pos_cache'][num_pos:]
        caches['neg_cache'] = caches['neg_cache'][num_pos:]

        #x_p = self.__get_random_positive_tiles(num_pos)
        #x_n = self.__get_random_negative_tiles(num_neg)

        y_p = np.ones((num_pos))
        y_n = np.zeros((num_neg))

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
        return x[p], y[p]
