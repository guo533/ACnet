from collections import OrderedDict

from torch.utils.data import Dataset
import cv2
import numpy as np

import hovernet_benchmark.utils as hb_utils
import hovernet_benchmark.process.targets as targets
from hovernet_benchmark.data.dataloader import FileLoader as FileLoaderHoVer
import albumentations as A

import sl.utils as utils

def gaussian_blur(images, random_state, parents, hooks, max_ksize=3):
    """Apply Gaussian blur to input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    ksize = random_state.randint(0, max_ksize, size=(2,))
    ksize = tuple((ksize * 2 + 1).tolist())

    ret = cv2.GaussianBlur(
        img, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REPLICATE
    )
    ret = np.reshape(ret, img.shape)
    ret = ret.astype(np.uint8)
    return [ret]


####
def median_blur(images, random_state, parents, hooks, max_ksize=3):
    """Apply median blur to input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    ksize = random_state.randint(0, max_ksize)
    ksize = ksize * 2 + 1
    ret = cv2.medianBlur(img, ksize)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_hue(images, random_state, parents, hooks, range=None):
    """Perturbe the hue of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    hue = random_state.uniform(*range)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if hsv.dtype.itemsize == 1:
        # OpenCV uses 0-179 for 8-bit images
        hsv[..., 0] = (hsv[..., 0] + hue) % 180
    else:
        # OpenCV uses 0-360 for floating point images
        hsv[..., 0] = (hsv[..., 0] + 2 * hue) % 360
    ret = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_saturation(images, random_state, parents, hooks, range=None):
    """Perturbe the saturation of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    value = 1 + random_state.uniform(*range)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret = img * value + (gray * (1 - value))[:, :, np.newaxis]
    ret = np.clip(ret, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_contrast(images, random_state, parents, hooks, range=None):
    """Perturbe the contrast of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    value = random_state.uniform(*range)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    ret = img * value + mean * (1 - value)
    ret = np.clip(img, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_brightness(images, random_state, parents, hooks, range=None):
    """Perturbe the brightness of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    value = random_state.uniform(*range)
    ret = np.clip(img + value, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]


def _gaussian_blur(*args):
    return gaussian_blur(*args, max_ksize=3)


def _median_blur(*args):
    return median_blur(*args, max_ksize=3)


def _add_to_hue(*args):
    return add_to_hue(*args, range=(-8, 8))


def _add_to_saturation(*args):
    return add_to_saturation(*args, range=(-0.2, 0.2))


def _add_to_brightness(*args):
    return add_to_brightness(*args, range=(-26, 26))


def _add_to_contrast(*args):
    return add_to_contrast(*args, range=(0.75, 1.25))

class FileLoaderExt(FileLoaderHoVer):

    def __init__(
            self,
            file_list,
            with_type=False,
            input_shape=None,
            mask_shape=None,
            mode="train",
            setup_augmentor=True,
            use_affine=True,
    ):
        super(FileLoaderExt, self).__init__(file_list, with_type=with_type, input_shape=input_shape,
                                            mask_shape=mask_shape, mode=mode, setup_augmentor=False,
                                            use_affine=use_affine)

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

        img = hb_utils.cropping_center(img, self.input_shape)
        feed_dict = {"img": img}

        inst_map = ann[..., 0]  # HW1 -> HW
        feed_dict['instance_map'] = inst_map
        # 只有类hover-net才能开启
        feed_dict.update(targets.gen_targets(inst_map, self.mask_shape))

        if self.with_type:
            type_map = (ann[..., 1]).copy()
            type_map = hb_utils.cropping_center(type_map, self.mask_shape)
            feed_dict["tp_map"] = type_map

            # contour map
            # contour_map = (ann[..., 2]).copy()
            # contour_map = hb_utils.cropping_center(contour_map, self.mask_shape)
            # feed_dict['cr_map'] = contour_map
        return feed_dict

    def get_test_data(self, idx):
        path = self.file_list[idx]
        if self.use_affine:
            data = np.load(path).astype('uint8')
            feed_dict = {"img": data}
        else:
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
        if self.mode == 'val':
            transforms = A.Compose([])
            return transforms



class FileLoader(Dataset):
    """Data Loader. Loads images from a file list and performs augmentation with the albumentation library.

    Parameters
    ----------
        file_list (list): list of filenames to load
        mode (str): 'train' or 'val'
    """
    def __init__(self, file_list, mode='train', num_types=6):
        self.mode = mode
        self.file_list = file_list
        self.num_types = num_types

    def get_training_data(self, idx):
        path = self.file_list[idx]
        data = np.load(path)
        # split stacked channel into image and label
        image = data[..., :3].astype('uint8')  # rgb image

        # During the training stage, instance_map is not used
        instance_map = data[..., 3].astype('int64')
        type_map = data[..., 4].astype('int64')
        type_map = utils.one_hot(type_map, num_classes=self.num_types)

        transforms = self.get_augmentation()
        transformed = transforms(image=image, mask=type_map)

        aug_type_map = transformed['mask']
        feed_dict = OrderedDict()
        feed_dict['image'] = transformed['image']
        feed_dict['type_map'] = aug_type_map
        return feed_dict

    def get_val_data(self, idx):
        path = self.file_list[idx]
        data = np.load(path)

        # split stacked channel into image and label
        image = data[..., :3].astype('uint8')

        instance_map = data[..., 3].astype('int64')
        type_map = data[..., 4].astype('int64')

        # During validation stage, augmentation cropping includes at most
        transforms = self.get_augmentation()
        transformed = transforms(image=image, masks=[instance_map, type_map])

        aug_instance_map = transformed['masks'][0]
        aug_type_map = transformed['masks'][1]

        feed_dict = OrderedDict()
        feed_dict['image'] = transformed['image']
        feed_dict['instance_map'] = aug_instance_map
        feed_dict['type_map'] = aug_type_map
        return feed_dict

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.get_training_data(idx)
        return self.get_val_data(idx)

    def get_augmentation(self):
        assert self.mode in ['train', 'val'], 'mode should be one of train or val, but now it is ' + self.mode
        if self.mode == 'train':
            transforms = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf([A.GaussianBlur(), A.MedianBlur(), A.GaussNoise()], p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10,
                                     val_shift_limit=10, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1,
                                           contrast_limit=0.1, p=0.5)
            ])
            return transforms
        if self.mode == 'val':
            transforms = A.Compose([])
            return transforms


class CellLabelsPathFileLoader(Dataset):

    def __init__(self, file_list, num_types=6):
        self.file_list = file_list
        self.num_types = num_types

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx][0])

        image = data[..., :3].astype('uint8')
        instance_map = data[..., 3].astype('int64')
        type_map = data[..., 4].astype('int64')

        inst_type_map = utils.create_ann_inst_type_map(instance_map, type_map, self.num_types, reserve_background=True)

        feed_dict = OrderedDict()
        feed_dict['image'] = image
        feed_dict['instance_map'] = instance_map
        feed_dict['inst_type_map'] = inst_type_map
        feed_dict['labels_path'] = self.file_list[idx][1]
        return feed_dict


class SubtypeMapFileLoader(Dataset):
    """Data Loader. Loads images from a file list and performs augmentation with the albumentation library.

    Parameters
    ----------
        file_list (list): list of filenames to load
        mode (str): 'train' or 'val'
    """
    def __init__(self, file_list, mode='train', num_types=6, num_subtypes=10, returns_subtype_maps=False):
        self.mode = mode
        self.file_list = file_list
        self.num_types = num_types
        self.num_subtypes = num_subtypes
        self.returns_subtype_maps = returns_subtype_maps

    def get_training_data(self, idx):
        path = self.file_list[idx]
        data = np.load(path)
        # split stacked channel into image and label
        image = data[..., :3].astype('uint8')  # rgb image

        # During the training stage, instance_map is not used
        instance_map = data[..., 3].astype('int64')
        type_map = data[..., 4].astype('int64')
        type_map = utils.one_hot(type_map, num_classes=self.num_types)

        transforms = self.get_augmentation()
        feed_dict = OrderedDict()
        if self.returns_subtype_maps:
            subtype_map = data[..., 7].astype('int64')
            # dead cells do not have subtypes
            subtype_map[subtype_map == 8] = 7
            subtype_map[subtype_map == 9] = 8
            subtype_map[subtype_map == 10] = 9
            subtype_map = utils.one_hot(subtype_map, num_classes=self.num_subtypes)
            transformed = transforms(image=image, masks=[type_map, subtype_map])
            feed_dict['subtype_map'] = transformed['masks'][1]
        else:
            transformed = transforms(image=image, masks=[type_map])
        feed_dict['image'] = transformed['image']
        feed_dict['type_map'] = transformed['masks'][0]
        return feed_dict

    def get_val_data(self, idx):
        path = self.file_list[idx]
        data = np.load(path)

        # split stacked channel into image and label
        image = data[..., :3].astype('uint8')

        instance_map = data[..., 3].astype('int64')
        type_map = data[..., 4].astype('int64')

        # During validation stage, augmentation cropping includes at most
        transforms = self.get_augmentation()
        transformed = transforms(image=image, masks=[instance_map, type_map])

        aug_instance_map = transformed['masks'][0]
        aug_type_map = transformed['masks'][1]

        feed_dict = OrderedDict()
        feed_dict['image'] = transformed['image']
        feed_dict['instance_map'] = aug_instance_map
        feed_dict['type_map'] = aug_type_map
        return feed_dict

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.get_training_data(idx)
        return self.get_val_data(idx)

    def get_augmentation(self):
        assert self.mode in ['train', 'val'], 'mode should be one of train or val, but now it is ' + self.mode
        if self.mode == 'train':
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
        if self.mode == 'val':
            transforms = A.Compose([])
            return transforms
