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

        real_img = None
        real_cond = None
        while real_img is None or real_cond is None:
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

        return img,real_cond,desired_cond

    def __len__(self):
        return self._dataset_size

    def _read_dataset_paths(self):
        self._imgs_dir = self._opt.dataset

        # read ids
        json_filenames = glob.glob(self._imgs_dir+'/*/*.json')
        self._ids = []
        # self._conds = []
        for filename in json_filenames:
            self._ids.append(filename.split('.')[0])
        if self._is_for_train:
            self._ids = self._ids[:-2000]
        else:
            self._ids = self._ids[-2000:]
        # dataset size
        self._dataset_size = len(self._ids)

    def _create_transform(self):
        if self._is_for_train:
            transform_list = [
                transforms.Resize((self._opt.image_size,self._opt.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                   std=[0.5, 0.5, 0.5]),
                              ]
        else:
            transform_list = [
                transforms.Resize((self._opt.image_size,self._opt.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                   std=[0.5, 0.5, 0.5]),
                              ]
        self._transform = transforms.Compose(transform_list)

    def _get_cond_by_id(self, id):
        filepath = id+'.json'
        with open(filepath) as file:
            _cond = json.load(file)
            return np.array(_cond,dtype=np.float32)

    def _get_img_by_id(self, id):
        filepath = id+'.jpg'
        return cv_utils.read_cv2_img(filepath), filepath

    def _generate_random_cond(self):
        return np.random.uniform(0, 1, [self._opt.cond_nc]).astype(np.float32)
