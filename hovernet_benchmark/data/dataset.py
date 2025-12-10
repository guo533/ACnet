

class BaseDataset(object):
    """Abstract class for interface of subsequent classes.
    Main idea is to encapsulate how each dataset should parse
    their images and annotations.
    For example:
        class CoNSePDataset(BaseDataset):


        def load_img(self, path):
            return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        def load_ann(self, path, with_type=False):
            # assumes that ann is HxW
            ann_inst = sio.loadmat(path)["inst_map"]
            if with_type:
                ann_type = sio.loadmat(path)["type_map"]

                # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
                # If own dataset is used, then the below may need to be modified
                ann_type[(ann_type == 3) | (ann_type == 4)] = 3
                ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

                ann = np.dstack([ann_inst, ann_type])
                ann = ann.astype("int32")
            else:
                ann = np.expand_dims(ann_inst, -1)
                ann = ann.astype("int32")
            return ann
    """

    def load_img(self, path):
        raise NotImplementedError

    def load_ann(self, path, with_type=False):
        raise NotImplementedError
