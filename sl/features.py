import math
import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure, feature, color
from scipy.ndimage.measurements import find_objects
from skimage.util import img_as_ubyte
import base.base_utils as the_base_utils

def interpolate_vector(vector, length):
    vector = the_base_utils.ensure_array(vector)
    vector = torch.tensor(vector)
    vector = vector.unsqueeze(0).unsqueeze(0)
    vector = F.interpolate(vector, size=(1, length))
    vector = vector.squeeze(0).squeeze(0)
    return the_base_utils.ensure_array(vector)



def extract_manual_features(image, inst_type_map, distances=(1,), angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4), epsilon=1e-8):
    # masks: 0 -> background, 1 -> type1, ...
    count = 0
    manual_features = np.zeros((len(np.unique(inst_type_map[..., 1:])[1:]), 3 + len(distances) * len(angles) * 2), dtype=np.float32)
    brightness = color.rgb2gray(image)
    background_map = inst_type_map[..., 0]
    avg_background_brightness = brightness[background_map != 0].sum() / (background_map.sum() + epsilon)
    tmp = np.log(avg_background_brightness / (brightness + epsilon))
    for type_id in range(1, inst_type_map.shape[2]):
        type_mask = inst_type_map[..., type_id]
        inst_id_list, inst_id_pixel_counts = np.unique(type_mask, return_counts=True)
        inst_id_list = inst_id_list[1:]
        inst_id_pixel_counts = inst_id_pixel_counts[1:]
        # manual feature 1: instance id with area
        inst_id_pixel_count_pairs = list(zip(inst_id_list, inst_id_pixel_counts))
        for inst_id, area in inst_id_pixel_count_pairs:
            single_cell_mask = (type_mask == inst_id).astype('int64')
            single_cell_contour = measure.find_contours(single_cell_mask)
            perimeter = epsilon
            for contour in single_cell_contour:
                perimeter += len(contour)
            # manual feature 2: nucleus roundness
            nucleus_roundness = (4 * math.pi * area) / perimeter

            # manual feature 3: integral optical density
            iod = (tmp * single_cell_mask).sum()

            objects = find_objects(type_mask)
            for i, sl in enumerate(objects, 1):
                if sl is None:
                    continue
                interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, type_mask.shape)]
                shrink_slice = _shrink(interior)
                single_cell_img_patch = image[_grow(sl, interior)]
                single_cell_mask_patch = type_mask[_grow(sl, interior)]
                glcm = feature.greycomatrix(img_as_ubyte(color.rgb2gray(single_cell_img_patch)), distances, angles)

                # manual feature 4: energy
                energy = feature.greycoprops(glcm, 'energy')
                # manual feature 5: contrast
                contrast = feature.greycoprops(glcm, 'contrast')

            cell_feature = np.concatenate([np.array([[area, nucleus_roundness, iod]]), energy, contrast], axis=1)
            manual_features[count] = cell_feature.flatten()
            count += 1
    return manual_features


def extract_deep_features(feature_maps, instance_map, dim):
    deep_features = np.zeros((instance_map.max(), dim))
    for idx in range(1, instance_map.max() + 1):
        locations = np.where(instance_map == idx)
        vectors = feature_maps[locations]
        vector = np.reshape(vectors, (1, vectors.shape[0] * vectors.shape[1]))
        vector = interpolate_vector(vector, dim)
        deep_features[idx - 1] = vector
    return deep_features


def extract_prototypes(feature_maps, type_map, epsilon=1e-8):
    """Extract prototypes of classes from feature maps
    Parameters
    ----------
    feature_maps (torch.Tensor): its dim is [B, H, W, C]
    type_map (torch.Tensor): its dim is [B, H, W, C]

    Returns
    -------
    prototypes (torch.Tensor): prototypes of classes, its dim is [B, C, num_classes]
    """
    num_types = type_map.shape[-1]
    prototypes = []
    for i in range(num_types):
        prototypes.append(torch.unsqueeze(torch.sum(feature_maps * torch.unsqueeze(type_map[..., i], -1), (1, 2)) / torch.unsqueeze(torch.sum(type_map[..., i], dim=(1, 2)) + epsilon, dim=-1), dim=-1))
    return torch.concat(prototypes, dim=-1)


def _shrink(interior):
    return tuple(slice(int(w[0]), (-1 if w[1] else None)) for w in interior)


def _grow(sl, interior):
    return tuple(slice(s.start - int(w[0]), s.stop + int(w[1])) for s, w in zip(sl, interior))
