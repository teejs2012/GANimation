import os.path
import torchvision.transforms as transforms
from data.dataset import DatasetBase
from PIL import Image
import random
import numpy as np
import pickle
from utils import cv_utils
import glob
import json

class AusDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(AusDataset, self).__init__(opt, is_for_train)
        self._name = 'AusDataset'
        self._is_for_train = is_for_train
        # read dataset
        self._read_dataset_paths()

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        # start_time = time.time()
        real_img = None
        real_cond = None
        while real_img is None or real_cond is None:
            # if sample randomly: overwrite index
            if not self._opt.serial_batches:
                index = random.randint(0, self._dataset_size - 1)

            # get sample data
            sample_id = self._ids[index]

            real_img, real_img_path = self._get_img_by_id(sample_id)
            real_cond = self._get_cond_by_id(sample_id)

            if real_img is None:
                print('error reading image %s, skipping sample' % sample_id)
            if real_cond is None:
                print('error reading aus %s, skipping sample' % sample_id)

        desired_cond = self._generate_random_cond()

        # transform data
        img = self._transform(Image.fromarray(real_img))

        # pack data
        sample = {'real_img': img,
                  'real_cond': real_cond,
                  'desired_cond': desired_cond,
                  'sample_id': sample_id,
                  'real_img_path': real_img_path
                  }

        # print (time.time() - start_time)

        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset_paths(self):
        self._root = self._opt.data_dir
        self._imgs_dir = os.path.join(self._root, self._opt.images_folder)

        # read ids
        # use_ids_filename = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        # use_ids_filepath = os.path.join(self._root, use_ids_filename)
        json_filenames = glob.glob(self._imgs_dir+'/*/*.json')
        self._ids = []
        # self._conds = []
        for filename in json_filenames:
            self._ids.append(filename.split('.')[0])
        if self._is_for_train:
            self._ids = self._ids[:-3]
        else:
            self._ids = self._ids[-3:]
        # dataset size
        self._dataset_size = len(self._ids)

    def _create_transform(self):
        if self._is_for_train:
            transform_list = [transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]),
                              ]
        else:
            transform_list = [transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]),
                              ]
        self._transform = transforms.Compose(transform_list)

    def _get_cond_by_id(self, id):
        filepath = id+'.json'
        with open(filepath) as file:
            _cond = json.load(file)
            return np.array(_cond)
        # if id in self._conds:
        #     return self._conds[id]
        # else:
        #     return None

    def _get_img_by_id(self, id):
        filepath = id+'.jpg'
        return cv_utils.read_cv2_img(filepath), filepath

    def _generate_random_cond(self):
        cond = None
        while cond is None:
            rand_sample_id = self._ids[random.randint(0, self._dataset_size - 1)]
            cond = self._get_cond_by_id(rand_sample_id)
            cond += np.random.uniform(-0.1, 0.1, cond.shape)
            cond = np.clip(cond,a_min=0,a_max=1)
        return cond
