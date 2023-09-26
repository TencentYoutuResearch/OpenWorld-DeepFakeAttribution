import os
import cv2
import pandas as pd

from torch.utils.data import Dataset
from loguru import logger

from owdfa.datasets.utils import dlib_crop_face

try:
    import dlib
except:
    print('Please install dlib when using face detection!')


class DFA(Dataset):
    def __init__(self,
                 data_root,
                 split='train',
                 meta_paths=None,
                 train_ratio_per_class=0.8,
                 seed=1234,
                 transform=None,
                 transform_twice=False,
                 crop_face=True,
                 predictor_path=None,
                 mod='all',
                 pseudo=False,
                 **kwargs
                 ):
        self.data_root = data_root
        self.split = split
        self.meta_paths = meta_paths
        self.train_ratio_per_class = train_ratio_per_class
        self.transform = transform
        self.transform_twice = transform_twice
        self.seed = seed
        self.crop_face = crop_face
        self.predictor_path = predictor_path
        self.mod = mod
        self.pseudo = pseudo

        self.setup()
        self.default_data = None

        if self.crop_face:
            logger.info("Using dlib for face detection!")
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.predictor_path)

    def setup(self):
        if type(self.meta_paths) is str:
            self.meta_paths = [self.meta_paths]

        train_dfs = []
        test_dfs = []
        for meta_path in self.meta_paths:
            assert os.path.exists(meta_path), f'{meta_path} not exists!'
            sub_df = pd.read_csv(meta_path)
            train_df, test_df = self._split_data(
                sub_df, self.train_ratio_per_class, self.seed)
            train_dfs.append(train_df)
            test_dfs.append(test_df)
        train_dfs = pd.concat(train_dfs, ignore_index=True)
        test_dfs = pd.concat(test_dfs, ignore_index=True)

        if self.pseudo:
            idx = train_dfs[train_dfs.tag == 2].index
            train_dfs.loc[idx, 'label'] = train_dfs.loc[idx, 'pseudo']

        self.items = train_dfs if self.split == 'train' else test_dfs

        logger.info(f'Total {self.split} numbers: {len(self.items)}')

    def _split_data(self, meta_df, train_ratio_per_class, seed):
        if self.mod == 'all':
            meta_df = meta_df[meta_df.tag != 0]
        elif self.mod == 'labeled':
            meta_df = meta_df[meta_df.tag == 1]
        elif self.mod == 'unlabeled':
            meta_df = meta_df[meta_df.tag == 2]
        else:
            assert False

        train_dfs = []
        for source in sorted(meta_df.image_source.unique()):
            for label in sorted(meta_df.label.unique()):
                train_df = meta_df[meta_df.image_source == source][meta_df.label == label].sample(
                    frac=train_ratio_per_class, random_state=seed)
                train_dfs.append(train_df)
        train_df = pd.concat(train_dfs)
        test_df = meta_df[~meta_df.index.isin(train_df.index)]

        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        return train_df, test_df

    def load_img(self, img_path):
        img = cv2.imread(img_path)

        if self.crop_face:
            # margin = random.uniform(1.0, 1.5) if self.split == 'train' else 1.2
            img = dlib_crop_face(img, self.detector,
                                 self.predictor, align=False, margin=1.2)

        img = img[:, :, ::-1]
        return img

    def get_method(self):
        return {label: method for method, label in
                self.items.label.groupby(self.items.method).value_counts().keys()}

    def get_labels(self):
        return self.items.label

    def get_tags(self):
        return self.items.tag

    def __getitem__(self, index):
        img_path, label, face_type, tag = self.items.loc[index, [
            'img_path', 'label', 'face_type', 'tag']]

        try:
            img = self.load_img(img_path)

            if self.transform_twice:
                img1 = self.transform(image=img)['image']
                img2 = self.transform(image=img)['image']
                img = [img1, img2]
            else:
                if self.transform is not None:
                    img = self.transform(image=img)['image']

            item = {'image': img, 'target': label, 'idx': index,
                    'tag': tag, 'face_type': face_type, 'img_path': img_path}

        except Exception as e:
            logger.info(f'{e}: {img_path}')
            return self.default_data

        if self.default_data is None:
            self.default_data = item

        return item

    def __len__(self):
        return len(self.items)

    def __exit__(self):
        if self.lmdb_dir is not None:
            self.lmdb.close()
