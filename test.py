import os
import argparse
import glob
import cv2
# from utils import face_utils
from utils import cv_utils
# import face_recognition
from PIL import Image
import torchvision.transforms as transforms
import torch
import pickle
import numpy as np
from models.models import ModelsFactory
from options.test_options import TestOptions

class MorphFacesInTheWild:
    def __init__(self, opt):
        self._opt = opt
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._model.set_eval()
        self._transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

    def morph_file(self, img_path):
        img = cv_utils.read_cv2_img(img_path)
        for i in range(10):
            param = i / 10
            print(param)
            cond = np.array([param, 1 - param])
            morphed_img, original_cond = self._morph_face(img, cond)
            print(original_cond)
            output_name = '%s_out_%d.png' % (os.path.basename(img_path),i)
            self._save_img(morphed_img, output_name)


    def _morph_face(self, face, expresion):
        face = torch.unsqueeze(self._transform(Image.fromarray(face)), 0)
        expresion = torch.unsqueeze(torch.from_numpy(expresion), 0)
        test_batch = {'real_img': face, 'real_cond': expresion, 'desired_cond': expresion, 'sample_id': torch.FloatTensor(), 'real_img_path': []}
        self._model.set_input(test_batch)
        imgs, data = self._model.predict()
        return imgs['concat'],data['real_cond']

    def _save_img(self, img, filename):
        filepath = os.path.join(self._opt.output_dir, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img)


def main():
    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    morph = MorphFacesInTheWild(opt)

    image_path = opt.input_path

    morph.morph_file(image_path)



if __name__ == '__main__':
    main()
