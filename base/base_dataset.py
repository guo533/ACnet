import numpy as np

from torch.utils.data import Dataset
import base.base_utils as utils
import base.targets as targets
import albumentations as A


class FileLoaderExt(Dataset):

    def __init__(self, file_list, with_type=False,
                 input_shape=None, mask_shape=None,
                 mode="train", gen_hv=True):
        super(FileLoaderExt, self).__init__()
        self.id = 0
        self.mode = mode
        self.with_type = with_type
        self.file_list = file_list
        self.mask_shape = mask_shape
        self.input_shape = input_shape
        self.gen_hv = gen_hv

    def get_training_data(self, idx):
        path = self.file_list[idx]
        data = np.load(path)
        # split stacked channel into image and label
        img = (data[..., :3]).astype("uint8")  # RGB images
        ann = (data[..., 3:]).astype("int32")

        transforms = self.get_augmentation()
        transformed = transforms(image=img, mask=ann)
        img = transformed['image']
        ann = transformed['mask']

        img = utils.cropping_center(img, self.input_shape)
        feed_dict = {"img": img}

        inst_map = ann[..., 0]
        if self.gen_hv:
            feed_dict.update(targets.gen_targets(inst_map, self.mask_shape))

        if self.with_type:
            type_map = (ann[..., 1]).copy()
            type_map = utils.cropping_center(type_map, self.mask_shape)
            feed_dict["tp_map"] = type_map
        return feed_dict

    def get_test_data(self, idx):
        path = self.file_list[idx]
        data = np.load(path)
        img = data[..., :3].astype('uint8')
        ann = data[..., 3:].astype('int32')
        feed_dict = {'img': img, 'ann': ann}
        return feed_dict

    def get_augmentation(self):
        if self.mode == "train":
            transforms = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf([A.GaussianBlur(), A.MedianBlur(), A.GaussNoise()], p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10,
                                     val_shift_limit=10, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1,
                                           contrast_limit=0.1, p=0.5),
            ])
            return transforms
        transforms = A.Compose([])
        return transforms
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.get_training_data(idx)
        return self.get_test_data(idx)


class LUSCTestFileLoaderExt(Dataset):

    def __init__(self, file_list, with_type=False, input_shape=None, mask_shape=None):
        super(LUSCTestFileLoaderExt, self).__init__()
        self.id = 0
        self.with_type = with_type
        self.file_list = file_list
        self.mask_shape = mask_shape
        self.input_shape = input_shape

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return self.get_test_data(idx)

    def get_test_data(self, idx):
        path = self.file_list[idx]
        data = np.load(path)
        img = data[..., :3].astype('uint8')
        ann = data[..., 3:].astype('int32')
        feed_dict = {'img': img, 'ann': ann}
        return feed_dict
