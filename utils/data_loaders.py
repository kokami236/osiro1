
import json
import torch
import logging
import numpy as np
import random
import torch.utils.data.dataset
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
import utils.data_transforms
from enum import Enum, unique
from tqdm import tqdm
from utils.io import IO
import os
import glob

label_mapping = {
    3: '03001627',
    6: '04379243',
    5: '04256520',
    1: '02933112',
    4: '03636649',
    2: '02958343',
    0: '02691156',
    7: '04530566'
}


@unique
class DatasetSubset(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


def collate_fn(batch):
    taxonomy_ids = []
    model_ids = []
    data = {}

    for sample in batch:
        taxonomy_ids.append(sample[0])
        model_ids.append(sample[1])
        _data = sample[2]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)

    return taxonomy_ids, model_ids, data


code_mapping = {
    'plane': '02691156',
    'cabinet': '02933112',
    'car': '02958343',
    'chair': '03001627',
    'lamp': '03636649',
    'couch': '04256520',
    'table': '04379243',
    'watercraft': '04530566',
}


def read_ply(file_path):
    pc = o3d.io.read_point_cloud(file_path)
    ptcloud = np.array(pc.points)
    return ptcloud


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.cache = dict()  # ここに point_count をキャッシュする

    def __len__(self):
        return len(self.file_list)

    def _get_point_count(self, path: str) -> int:
        # 点数だけ欲しいので、IO.getのshape[0]をキャッシュ
        if path in self.cache:
            return self.cache[path]
        try:
            pts = IO.get(path)
            n = int(pts.shape[0])
        except Exception:
            n = -1
        self.cache[path] = n
        return n

    def _select_best_view_path(self, paths, shuffle: bool) -> str:
        """
        paths: partialの候補（list）
        - min_keep 以上のものだけ残す
        - 残った中で点数が多い順に top_k を取り、その中から（shuffleなら）ランダム選択
        - 全滅なら “最大点数” を1つ選ぶ（保険）
        """
        sel = self.options.get('view_select', None)
        if sel is None:
            # 従来通り（ランダム or 先頭）
            return random.choice(paths) if shuffle else paths[0]

        min_keep = int(sel.get('min_keep', 256))
        top_k    = int(sel.get('top_k', 2))
        mode     = sel.get('mode', 'topk_random')  # 'topk_random' or 'max'

        # 各viewの点数を取得
        counts = [(p, self._get_point_count(p)) for p in paths]
        counts = [(p, c) for (p, c) in counts if c > 0]

        if len(counts) == 0:
            return paths[0]

        # min_keepでフィルタ
        good = [(p, c) for (p, c) in counts if c >= min_keep]

        # 全滅なら最大点数で保険
        if len(good) == 0:
            p_best = max(counts, key=lambda x: x[1])[0]
            return p_best

        # 点数で降順ソートしてTop-K
        good.sort(key=lambda x: x[1], reverse=True)
        topk = good[:max(1, min(top_k, len(good)))]

        if mode == 'max' or (not shuffle):
            # val/testや shuffle=False は最大を固定で使う（ブレない）
            return topk[0][0]

        # train は topk の中からランダム（多様性）
        return random.choice([p for (p, _) in topk])

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}

        shuffle = bool(self.options.get('shuffle', False))

        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]

            # ★ここが肝：partialの候補が list のとき view選択する
            if isinstance(file_path, list):
                if ri == 'partial_cloud':
                    file_path = self._select_best_view_path(file_path, shuffle=shuffle)
                else:
                    # gtcloud側がlistになることは通常ないが、念のため従来通り
                    file_path = random.choice(file_path) if shuffle else file_path[0]

            data[ri] = IO.get(file_path).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], data



class ShapeNetDataLoader(object):
    """
    PCN dataset: get dataset file list
    """
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.SHAPENET.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        n_renderings = self.cfg.DATASETS.SHAPENET.N_RENDERINGS if subset == DatasetSubset.TRAIN else 1
        file_list = self._get_file_list(self.cfg, self._get_subset(subset),
                                        n_renderings)
        transforms = self._get_transforms(self.cfg, subset)
        return Dataset(
            {
                'n_renderings': n_renderings,
                'required_items': ['partial_cloud', 'gtcloud'],
                'shuffle': subset == DatasetSubset.TRAIN
            }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback':
                'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback':
                'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback':
                'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        # Collect file list
        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' %
                         (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):

                if subset == 'test':
                    gt_path = cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (
                        subset, dc['taxonomy_id'], s)
                    
                    # 修正: 8つの視点すべてをテスト対象としてリストに追加する
                    # train_pcn.py で設定した PARTIAL_POINTS_PATH (%02dを含む) を使用
                    for i in range(cfg.DATASETS.SHAPENET.N_RENDERINGS):
                        file_list.append({
                            'taxonomy_id':
                            dc['taxonomy_id'],
                            'model_id':
                            s,
                            'partial_cloud_path':
                            cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (subset, dc['taxonomy_id'], s, i),
                            'gtcloud_path':
                            gt_path
                        })
                else:
                    file_list.append({
                        'taxonomy_id':
                        dc['taxonomy_id'],
                        'model_id':
                        s,
                        'partial_cloud_path': [
                            cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH %
                            (subset, dc['taxonomy_id'], s, i)
                            for i in range(n_renderings)
                        ],
                        'gtcloud_path':
                        cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH %
                        (subset, dc['taxonomy_id'], s),
                    })

        logging.info(
            'Complete collecting files of the dataset. Total files: %d' %
            len(file_list))
        return file_list


class ShapeNetCarsDataLoader(ShapeNetDataLoader):
    """
    ShapeNet only on car category
    """
    def __init__(self, cfg):
        super(ShapeNetCarsDataLoader, self).__init__(cfg)

        # Remove other categories except cars
        self.dataset_categories = [
            dc for dc in self.dataset_categories
            if dc['taxonomy_id'] == '02958343'
        ]


class Completion3DDataLoader(object):
    """
    Completion3D: get dataset file list
    """
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = [
            'partial_cloud'
        ] if subset == DatasetSubset.TEST else ['partial_cloud', 'gtcloud']

        return Dataset(
            {
                'required_items': required_items,
                'shuffle': subset == DatasetSubset.TRAIN
            }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': cfg.CONST.N_INPUT_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback':
                'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback':
                'ScalePoints',
                'parameters': {
                    'scale': 0.85
                },
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback':
                'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        elif subset == DatasetSubset.VAL:
            return utils.data_transforms.Compose([{
                'callback':
                'ScalePoints',
                'parameters': {
                    'scale': 0.85
                },
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback':
                'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'ToTensor',
                'objects': ['partial_cloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        # Collect file list
        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' %
                         (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_cloud_path':
                    cfg.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH %
                    (subset, dc['taxonomy_id'], s),
                    'gtcloud_path':
                    cfg.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH %
                    (subset, dc['taxonomy_id'], s),
                })

        logging.info(
            'Complete collecting files of the dataset. Total files: %d' %
            len(file_list))
        return file_list


class Completion3DPCCTDataLoader(Completion3DDataLoader):
    """
    Dataset Completion3D containing only plane, car, chair, table
    """
    def __init__(self, cfg):
        super(Completion3DPCCTDataLoader, self).__init__(cfg)

        # Remove other categories except couch, chairs, car, lamps
        cat_set = {'02691156', '03001627', '02958343',
                   '04379243'}  # plane, chair, car, table
        # cat_set = {'04256520', '03001627', '02958343', '03636649'}
        self.dataset_categories = [
            dc for dc in self.dataset_categories
            if dc['taxonomy_id'] in cat_set
        ]


class KittiDataLoader(object):
    """
    KITTI: extracted car objects
    """
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.KITTI.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ['partial_cloud', 'bounding_box']

        return Dataset({
            'required_items': required_items,
            'shuffle': False
        }, file_list, transforms)

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_transforms(self, cfg, subset):
        return utils.data_transforms.Compose([{
            'callback':
            'NormalizeObjectPose',
            'parameters': {
                'input_keys': {
                    'ptcloud': 'partial_cloud',
                    'bbox': 'bounding_box'
                }
            },
            'objects': ['partial_cloud', 'bounding_box']
        }, {
            'callback': 'RandomSamplePoints',
            'parameters': {
                'n_points': cfg.CONST.N_INPUT_POINTS
            },
            'objects': ['partial_cloud']
        }, {
            'callback':
            'ToTensor',
            'objects': ['partial_cloud', 'bounding_box']
        }])

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        # Collect file list
        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' %
                         (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_cloud_path':
                    cfg.DATASETS.KITTI.PARTIAL_POINTS_PATH % s,
                    'bounding_box_path':
                    cfg.DATASETS.KITTI.BOUNDING_BOX_FILE_PATH % s,
                })

        logging.info(
            'Complete collecting files of the dataset. Total files: %d' %
            len(file_list))
        return file_list


# ShapeNet-55/34
# Ref: https://github.com/yuxumin/PoinTr/blob/master/datasets/ShapeNet55Dataset.py
class ShapeNet55Dataset(torch.utils.data.dataset.Dataset):
    """
    ShapeNet55 dataset: return complete clouds, partial clouds are generated online
    """
    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.cache = dict()

    def __len__(self):
        return len(self.file_list)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}

        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            data[ri] = IO.get(file_path).astype(np.float32)
            # shapenet55
            data[ri] = self.pc_norm(data[ri])
            data[ri] = torch.from_numpy(data[ri]).float()

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], data


class ShapeNet55DataLoader(object):
    """
    ShapeNet55: get dataset file list
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = None
        return ShapeNet55Dataset(
            {
                'required_items': ['gtcloud'],
                'shuffle': subset == DatasetSubset.TRAIN
            }, file_list, transforms)

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset, n_renderings=1):
        file_list = []
        for dc in self.dataset_categories:
            samples = dc.get(subset, [])
            for s in tqdm(samples, leave=False):
                gt_path = cfg.DATASETS.CUSTOM.COMPLETE_POINTS_PATH % s

            # 候補viewを作る（存在チェック込み）
                partial_paths = []
                for i in range(cfg.DATASETS.CUSTOM.N_RENDERINGS):
                    p = cfg.DATASETS.CUSTOM.PARTIAL_POINTS_PATH % (s, i)
                    if os.path.exists(p):
                        partial_paths.append(p)

                if subset == 'test':
                # ★testは「存在するviewだけ」列挙
                    for p in partial_paths:
                        file_list.append({
                        'taxonomy_id': dc['taxonomy_id'],
                        'model_id': s,
                        'partial_cloud_path': p,
                        'gtcloud_path': gt_path
                    })
                else:
                # train/valは「リスト」で渡す（Dataset側が選ぶ）
                    file_list.append({
                    'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_cloud_path': partial_paths,
                    'gtcloud_path': gt_path
                })
        logging.info('Custom dataset: total files = %d', len(file_list))
        return file_list


#####################################追加
class CustomDataLoader(ShapeNetDataLoader):
    def __init__(self, cfg):
        self.cfg = cfg
        with open(cfg.DATASETS.CUSTOM.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        n_renderings = self.cfg.DATASETS.CUSTOM.N_RENDERINGS  # 5

        if subset == DatasetSubset.TRAIN:
            shuffle = True
            view_select = {'min_keep': 256, 'top_k': 2, 'mode': 'topk_random'}
        else:
            shuffle = False
            view_select = {'min_keep': 256, 'top_k': 1, 'mode': 'max'}

        file_list = self._get_file_list(self.cfg, self._get_subset(subset), n_renderings)
        transforms = self._get_transforms(self.cfg, subset)

        return Dataset(
            {
                'n_renderings': n_renderings,
                'required_items': ['partial_cloud', 'gtcloud'],
                'shuffle': shuffle,
                'view_select': view_select,
            },
            file_list,
            transforms
        )

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.CUSTOM.N_POINTS,
                    'jitter_sigma': 0.005,
                    'jitter_clip':  0.02,
                    'deterministic': False,
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.CUSTOM.N_POINTS,
                    'deterministic': True,
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset, n_renderings=1):
        file_list = []
        MIN_VIEWS_PER_SAMPLE = 2  # ★B: 2 view以上

        for dc in self.dataset_categories:
            samples = dc.get(subset, [])
            for s in tqdm(samples, leave=False):
                gt_path = cfg.DATASETS.CUSTOM.COMPLETE_POINTS_PATH % s
                if not os.path.exists(gt_path):
                    continue

                # 存在するpartialだけ集める
                partial_paths = []
                for i in range(n_renderings):
                    p = cfg.DATASETS.CUSTOM.PARTIAL_POINTS_PATH % (s, i)
                    if os.path.exists(p):
                        partial_paths.append(p)

                # ★B条件
                if len(partial_paths) < MIN_VIEWS_PER_SAMPLE:
                    continue

                if subset == 'test':
                    # testは存在するviewだけ列挙
                    for p in partial_paths:
                        file_list.append({
                            'taxonomy_id': dc['taxonomy_id'],
                            'model_id': s,
                            'partial_cloud_path': p,
                            'gtcloud_path': gt_path
                        })
                else:
                    # train/val は list で渡す（Dataset側でview選択）
                    file_list.append({
                        'taxonomy_id': dc['taxonomy_id'],
                        'model_id': s,
                        'partial_cloud_path': partial_paths,
                        'gtcloud_path': gt_path
                    })

        logging.info('Custom dataset: total files = %d', len(file_list))
        return file_list

# //////////////////////////////////////////// = Dataset Loader Mapping = //////////////////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'Completion3D': Completion3DDataLoader,
    'Completion3DPCCT': Completion3DPCCTDataLoader,
    'ShapeNet': ShapeNetDataLoader,
    'ShapeNetCars': ShapeNetCarsDataLoader,
    'KITTI': KittiDataLoader,
    'ShapeNet55': ShapeNet55DataLoader,
    'Custom': CustomDataLoader,
}  # yapf: disable
