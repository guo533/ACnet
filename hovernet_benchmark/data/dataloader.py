import numpy as np
from torch.utils.data import Dataset

from imgaug import augmenters as iaa

import hovernet_benchmark.utils as utils
import hovernet_benchmark.process.targets as targets

from hovernet_benchmark.data.augs import (
    add_to_brightness,
    add_to_contrast,
    add_to_hue,
    add_to_saturation,
    gaussian_blur,
    median_blur,
)


class FileLoader(Dataset):
    """Data Loader. Loads images from a file list and
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are
    generated.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w]
        mask_shape: shape of the output [h,w]
        mode: 'train' or 'test'
    """
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
        assert input_shape is not None and mask_shape is not None

        self.id = 0
        self.mode = mode
        self.with_type = with_type
        self.file_list = file_list
        self.mask_shape = mask_shape
        self.input_shape = input_shape
        self.use_affine = use_affine

        self.shape_augs = None
        self.input_augs = None
        if setup_augmentor:
            self.setup_augmentor(0, 0)

    def setup_augmentor(self, worker_id, seed):
        augmentor = self.get_augmentation(seed)
        self.shape_augs = iaa.Sequential(augmentor[0])
        self.input_augs = iaa.Sequential(augmentor[1])
        self.id = self.id + worker_id

    def get_training_data(self, idx):
        path = self.file_list[idx]
        data = np.load(path)
        # split stacked channel into image and label
        img = (data[..., :3]).astype("uint8")  # RGB images
        if data.shape[-1] == 3:
            img = utils.cropping_center(img, self.input_shape)
            feed_dict = {"img": img}
            return feed_dict
        ann = (data[..., 3:]).astype("int32")  # instance ID map

        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)
            ann = shape_augs.augment_image(ann)

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img = input_augs.augment_image(img)

        img = utils.cropping_center(img, self.input_shape)
        feed_dict = {"img": img}

        inst_map = ann[..., 0]  # HW1 -> HW
        if self.with_type:
            type_map = (ann[..., 1]).copy()
            type_map = utils.cropping_center(type_map, self.mask_shape)
            # type_map[type_map == 5] = 1  # merge neoplastic and non-neoplastic
            feed_dict["tp_map"] = type_map
        # 类似hovernet结构才需要
        # feed_dict.update(targets.gen_targets(inst_map, self.mask_shape))
        return feed_dict

    def get_test_data(self, idx):
        path = self.file_list[idx]
        if self.use_affine:
            data = np.load(path).astype('uint8')
            feed_dict = {"img": data}
        else:
            data = np.load(path)
            img = data[..., :3].astype('uint8')
            if data.shape[-1] == 3:
                feed_dict = {"img": img}
                return feed_dict
            ann = data[..., 3:].astype('int32')
            feed_dict = {'img': img, 'ann': ann}
        return feed_dict

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.get_training_data(idx)
        return self.get_test_data(idx)

    def get_augmentation(self, rng):
        shape_augs = []
        input_augs = []
        if self.mode == "train":
            shape_augs = [
                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate by -A to +A percent (per axis)
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    rotate=(-179, 179),  # rotate by -179 to +179 degrees
                    order=0,  # use nearest neighbour
                    backend="cv2",  # opencv for fast processing
                    seed=rng,
                ),
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                ),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
                iaa.Affine(rotate=90, backend="cv2", order=0, seed=rng)
            ]

            if not self.use_affine:
                shape_augs.pop(0)

            input_augs = [
                iaa.OneOf(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=_gaussian_blur,
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=_median_blur,
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                    ]
                ),
                iaa.Sequential(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=_add_to_hue,
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=_add_to_saturation,
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=_add_to_brightness,
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=_add_to_contrast,
                        ),
                    ],
                    random_order=True,
                ),
            ]
        return shape_augs, input_augs


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
