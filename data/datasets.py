import os, glob
import torch, sys
sys.path.append('/root/data1/wangxiaolin/Unet-RegSeg/data')
from torch.utils.data import Dataset
# from .data_utils import pkload
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from convert_labels3D import convert
import nibabel as nib

def load_array_if_path(var, load_as_numpy=True):
    """If var is a string and load_as_numpy is True, this function loads the array writen at the path indicated by var.
    Otherwise it simply returns var as it is."""
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(var), 'No such path: %s' % var
        var = np.load(var)
    return var
    
def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of an array.
    The number of dimensions is assumed to be the length of the shape, as long as the shape of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
    example 1: get_dims([150, 150, 150], max_channels=10) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=10) = (3, 3)
    example 3: get_dims([150, 150, 150, 15], max_channels=10) = (4, 1), because 5>3"""
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels

def reformat_to_list(var, length=None, load_as_numpy=False, dtype=None):
    """This function takes a variable and reformat it into a list of desired
    length and type (int, float, bool, str).
    If variable is a string, and load_as_numpy is True, it will be loaded as a numpy array.
    If variable is None, this function returns None.
    :param var: a str, int, float, list, tuple, or numpy array
    :param length: (optional) if var is a single item, it will be replicated to a list of this length
    :param load_as_numpy: (optional) whether var is the path to a numpy array
    :param dtype: (optional) convert all item to this type. Can be 'int', 'float', 'bool', or 'str'
    :return: reformatted list
    """

    # convert to list
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    if isinstance(var, (int, float, np.int, np.int32, np.int64, np.float, np.float32, np.float64)):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = [var[0]]
        else:
            var = np.squeeze(var).tolist()
    elif isinstance(var, str):
        var = [var]
    elif isinstance(var, bool):
        var = [var]
    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError('if var is a list/tuple/numpy array, it should be of length 1 or {0}, '
                                 'had {1}'.format(length, var))
    else:
        raise TypeError('var should be an int, float, tuple, list, numpy array, or path to numpy array')

    # convert items type
    if dtype is not None:
        if dtype == 'int':
            var = [int(v) for v in var]
        elif dtype == 'float':
            var = [float(v) for v in var]
        elif dtype == 'bool':
            var = [bool(v) for v in var]
        elif dtype == 'str':
            var = [str(v) for v in var]
        else:
            raise ValueError("dtype should be 'str', 'float', 'int', or 'bool'; had {}".format(dtype))
    return var

def find_closest_number_divisible_by_m(n, m, answer_type='lower'):
    """Return the closest integer to n that is divisible by m. answer_type can either be 'closer', 'lower' (only returns
    values lower than n), or 'higher' (only returns values higher than m)."""
    if n % m == 0:
        return n
    else:
        q = int(n / m)
        lower = q * m
        higher = (q + 1) * m
        if answer_type == 'lower':
            return lower
        elif answer_type == 'higher':
            return higher
        elif answer_type == 'closer':
            return lower if (n - lower) < (higher - n) else higher
        else:
            raise Exception('answer_type should be lower, higher, or closer, had : %s' % answer_type)

def pad_volume(volume, padding_shape, padding_value=0, aff=None, return_pad_idx=False):
    """Pad volume to a given shape
    :param volume: volume to be padded
    :param padding_shape: shape to pad volume to. Can be a number, a sequence or a 1d numpy array.
    :param padding_value: (optional) value used for padding
    :param aff: (optional) affine matrix of the volume
    :param return_pad_idx: (optional) the pad_idx corresponds to the indices where we should crop the resulting
    padded image (ie the output of this function) to go back to the original volume (ie the input of this function).
    :return: padded volume, and updated affine matrix if aff is not None.
    """

    # get info
    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, n_channels = get_dims(vol_shape)
    padding_shape = reformat_to_list(padding_shape, length=n_dims, dtype='int')

    # check if need to pad
    if np.any(np.array(padding_shape, dtype='int32') > np.array(vol_shape[:n_dims], dtype='int32')):

        # get padding margins
        min_margins = np.maximum(np.int32(np.floor((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)), 0)
        max_margins = np.maximum(np.int32(np.ceil((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)), 0)
        pad_idx = np.concatenate([min_margins, min_margins + np.array(vol_shape[:n_dims])])
        pad_margins = tuple([(min_margins[i], max_margins[i]) for i in range(n_dims)])
        if n_channels > 1:
            pad_margins = tuple(list(pad_margins) + [(0, 0)])

        # pad volume
        new_volume = np.pad(new_volume, pad_margins, mode='constant', constant_values=padding_value)

        if aff is not None:
            if n_dims == 2:
                min_margins = np.append(min_margins, 0)
            aff[:-1, -1] = aff[:-1, -1] - aff[:-1, :-1] @ min_margins

    else:
        pad_idx = np.concatenate([np.array([0] * n_dims), np.array(vol_shape[:n_dims])])

    # sort outputs
    output = [new_volume]
    if aff is not None:
        output.append(aff)
    if return_pad_idx:
        output.append(pad_idx)
    return output[0] if len(output) == 1 else tuple(output)

def n_score_normalize(data):
    """
    Normalize as n-score and minus minimum
    """
    mean = data.mean()
    std = np.std(data)
    data = (data - mean) / std
    return data - data.min()

def normalize_image_intensity(image):
    """
    对输入图像进行强度归一化处理。
    
    :param image: 待处理的图像数组。
    :return: 强度归一化后的图像数组。
    """
    # 将图像数据转换为浮点数，以便进行除法运算
    image = image.astype(np.float32)
    
    # 计算图像的最小和最大强度值
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    
    # 执行归一化：(image - min) / (max - min)
    # 结果中的所有值都将位于[0, 1]区间
    normalized_image = (image - min_intensity) / (max_intensity - min_intensity)
    
    return normalized_image

def rescale_volume(volume, new_min=0, new_max=255, min_percentile=2, max_percentile=98, use_positive_only=False):
    """This function linearly rescales a volume between new_min and new_max.
    :param volume: a numpy array
    :param new_min: (optional) minimum value for the rescaled image.
    :param new_max: (optional) maximum value for the rescaled image.
    :param min_percentile: (optional) percentile for estimating robust minimum of volume (float in [0,...100]),
    where 0 = np.min
    :param max_percentile: (optional) percentile for estimating robust maximum of volume (float in [0,...100]),
    where 100 = np.max
    :param use_positive_only: (optional) whether to use only positive values when estimating the min and max percentile
    :return: rescaled volume
    """

    # select only positive intensities
    new_volume = volume.copy()
    intensities = new_volume[new_volume > 0] if use_positive_only else new_volume.flatten()

    # define min and max intensities in original image for normalisation
    robust_min = np.min(intensities) if min_percentile == 0 else np.percentile(intensities, min_percentile)
    robust_max = np.max(intensities) if max_percentile == 100 else np.percentile(intensities, max_percentile)

    # trim values outside range
    new_volume = np.clip(new_volume, robust_min, robust_max)

    # rescale image
    if robust_min != robust_max:
        return new_min + (new_volume - robust_min) / (robust_max - robust_min) * (new_max - new_min)
    else:  # avoid dividing by zero
        return np.zeros_like(new_volume)

def load_volfile(datafile):
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % datafile

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nib' not in sys.modules:
            try:
                import nibabel as nib
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_data()

    else:  # npz
        X = np.load(datafile)['vol_data']

    return X

class TrainDataset(Dataset):
    def __init__(self, data_path, transforms=None):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        if len(self.paths[index]) == 5:
            tgt_path, src_path, _, s_seg_path, _ = self.paths[index]
        if len(self.paths[index]) == 4:
            tgt_path, src_path, s_seg_path, _ = self.paths[index]
        tgt, src = load_volfile(tgt_path), load_volfile(src_path)
        s_seg = load_volfile(s_seg_path)
        s_seg = convert(s_seg)

        # tgt = ndimage.zoom(tgt, (128 / 160, 128 / 192, 128 / 224), order=1)
        # src = ndimage.zoom(src, (128 / 160, 128 / 192, 128 / 224), order=1)
        # s_seg = ndimage.zoom(s_seg, (128 / 160, 128 / 192, 128 / 224), order=0)

        # tgt = ndimage.zoom(tgt, (160 / 160, 160 / 192, 160 / 224), order=1)
        # src = ndimage.zoom(src, (160 / 160, 160 / 192, 160 / 224), order=1)
        # s_seg = ndimage.zoom(s_seg, (160 / 160, 160 / 192, 160 / 224), order=0)

        tgt, src = tgt[None, ...], src[None, ...]  #(1, 160, 160, 160) (1, 160, 160, 160)
        s_seg = s_seg[None, ...]
        # tgt, src = self.transforms([tgt, src])
        # ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组
        tgt = np.ascontiguousarray(tgt)# [Bsize,channelsHeight,,Width,Depth]   
        src = np.ascontiguousarray(src)
        s_seg = np.ascontiguousarray(s_seg)
        tgt, src = torch.from_numpy(tgt), torch.from_numpy(src)
        return src, tgt, s_seg

    def __len__(self):
        return len(self.paths)
class VolWithContoursDataset(Dataset):
    
    def __init__(self, train_vol_names):
        self.train_vol_names = train_vol_names
        '''class torchvision.transforms.ToTensor'''

    def __getitem__(self, i):
        # Source_image, Source_seg, Target_image, Source_contours = self.train_vol_names[i]
        # image_0, image_1 = load_volfile(Target_image.replace('/root/data', '/temp4')), load_volfile(Source_image.replace('/root/data', '/temp4'))
        # s_seg, s_contours = load_volfile(Source_seg.replace('/root/data', '/temp4')), load_volfile(Source_contours.replace('/root/data', '/temp4'))
        
        Target_path = self.train_vol_names[i]
        # Target_path, _, _, _ = self.train_vol_names[i]
        Source_path = '/root/data1/wangxiaolin/RegSeg-main/RegSeg/DATA3.15/atlas_vol.nii'
        Source_seg = '/root/data1/wangxiaolin/RegSeg-main/RegSeg/DATA3.15/atlas_seg_14labels.nii'
        tgt, src = load_volfile(Target_path), load_volfile(Source_path)
        # s_seg, s_contours = load_volfile(Source_seg), load_volfile(Source_contours)
        s_seg = load_volfile(Source_seg)
        
        # import pdb; pdb.set_trace()
        # 新加的
        tgt = ndimage.zoom(tgt, (128 / 160, 128 / 192, 128 / 224), order=1)
        src = ndimage.zoom(src, (128 / 160, 128 / 192, 128 / 224), order=1)
        s_seg = ndimage.zoom(s_seg, (128 / 160, 128 / 192, 128 / 224), order=0)
        # s_contours = ndimage.zoom(s_contours, (128 / 160, 128 / 192, 128 / 224), order=0)

        # image_0 = ndimage.zoom(image_0, (160 / 160, 160 / 192, 160 / 224), order=1)
        # image_1 = ndimage.zoom(image_1, (160 / 160, 160 / 192, 160 / 224), order=1)
        # s_seg = ndimage.zoom(s_seg, (160 / 160, 160 / 192, 160 / 224), order=0)
        # s_contours = ndimage.zoom(s_contours, (160 / 160, 160 / 192, 160 / 224), order=0)

        tgt = torch.Tensor(tgt).float()
        tgt = tgt.unsqueeze(0)
        # image_0 = image_0.permute(0, 2, 3, 1)
        src = torch.Tensor(src).float()
        src = src.unsqueeze(0)
        # image_1 = image_1.permute(0, 2, 3, 1)

        s_seg = convert(s_seg)
        s_seg = torch.Tensor(s_seg).float()
        s_seg = s_seg.unsqueeze(0)
        
        # s_contours = convert(s_contours)
        # contours = torch.Tensor(s_contours).float()
        # contours = contours.unsqueeze(0)

        return tgt, src, s_seg
        # return tgt, src

    def __len__(self):
        return len(self.train_vol_names)
        
class TrainSegFakeDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        if len(self.paths[index]) == 4:
            tgt_path, src_path, tseg_path, sseg_path = self.paths[index]
        else:
            tgt_path, src_path, tseg_path, sseg_path, _ = self.paths[index]
        tgt, src = load_volfile(tgt_path), load_volfile(src_path)
        tgt_seg, src_seg = load_volfile(tseg_path), load_volfile(sseg_path)  
        tgt_seg = convert(tgt_seg)
        src_seg = convert(src_seg)

        # tgt = ndimage.zoom(tgt, (160 / 160, 160 / 192, 160 / 224), order=1)
        # src = ndimage.zoom(src, (160 / 160, 160 / 192, 160 / 224), order=1)

        # tgt_seg = ndimage.zoom(tgt_seg, (160 / 160, 160 / 192, 160 / 224), order=0)
        # src_seg = ndimage.zoom(src_seg, (160 / 160, 160 / 192, 160 / 224), order=0)

        tgt = ndimage.zoom(tgt, (128 / 160, 128 / 192, 128 / 224), order=1)
        src = ndimage.zoom(src, (128 / 160, 128 / 192, 128 / 224), order=1)

        tgt_seg = ndimage.zoom(tgt_seg, (128 / 160, 128 / 192, 128 / 224), order=0)
        src_seg = ndimage.zoom(src_seg, (128 / 160, 128 / 192, 128 / 224), order=0)

        tgt, src = tgt[None, ...], src[None, ...]
        tgt_seg, src_seg= tgt_seg[None, ...], src_seg[None, ...]

        # tgt, tgt_seg = self.transforms([tgt, tgt_seg])
        # src, src_seg = self.transforms([src, src_seg])

        tgt = np.ascontiguousarray(tgt)# [Bsize,channelsHeight,,Width,Depth]
        src = np.ascontiguousarray(src)
        tgt_seg = np.ascontiguousarray(tgt_seg)  # [Bsize,channelsHeight,,Width,Depth]
        src_seg = np.ascontiguousarray(src_seg)

        tgt, src, tgt_seg, src_seg = torch.from_numpy(tgt), torch.from_numpy(src), torch.from_numpy(tgt_seg), torch.from_numpy(src_seg)
        return src, tgt, src_seg, tgt_seg

    def __len__(self):
        return len(self.paths)

class TrainOneShot(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        # tgt_path, src_path, sseg_path, _ = self.paths[index]
        # src_path = '/root/data1/wangxiaolin/Unet-RegSeg/Data/ISeg/subject-4-T1.nii.gz'
        # sseg_path = '/root/data1/wangxiaolin/Unet-RegSeg/Data/ISeg/subject-4-label.nii.gz'
        # src_path = '/root/data1/wangxiaolin/Unet-RegSeg/Data/SKI10/9256759_vol.nii.gz'
        # sseg_path = '/root/data1/wangxiaolin/Unet-RegSeg/Data/SKI10/9256759_seg_4.nii.gz'
        src_path = '/root/data1/wangxiaolin/Unet-RegSeg/Data/t2_zoomed/subject-4-T2.nii.gz'
        sseg_path = '/root/data1/wangxiaolin/Unet-RegSeg/Data/t2_zoomed/subject-4-label.nii.gz'
        # src_path = '/user77/wangxiaolin/Unet-RegSeg/Data/atlas/atlas_vol.nii'
        # sseg_path = '/user77/wangxiaolin/Unet-RegSeg/Data/atlas/atlas_seg_14labels.nii'
        tgt_path = self.paths[index]
        tgt, src = load_volfile(tgt_path), load_volfile(src_path)
        tgt, src = normalize_image_intensity(tgt), normalize_image_intensity(src)
        src_seg = load_volfile(sseg_path)  
        # src_seg = convert(src_seg)

        tgt = ndimage.zoom(tgt, (128 / 160, 128 / 192, 128 / 224), order=1)
        src = ndimage.zoom(src, (128 / 160, 128 / 192, 128 / 224), order=1)

        src_seg = ndimage.zoom(src_seg, (128 / 160, 128 / 192, 128 / 224), order=0)

        tgt, src = tgt[None, ...], src[None, ...]
        # src_seg = self.to_categorical(src_seg)
        src_seg= src_seg[None, ...]

        tgt = np.ascontiguousarray(tgt)# [Bsize,channelsHeight,,Width,Depth]
        src = np.ascontiguousarray(src)
        src_seg = np.ascontiguousarray(src_seg)

        tgt, src, src_seg = torch.from_numpy(tgt), torch.from_numpy(src), torch.from_numpy(src_seg)
        # 返回顺序是source,target,source_seg
        return src, tgt, src_seg

    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((num_classes, n))
        categorical[y, np.arange(n)] = 1
        output_shape = (num_classes,) + input_shape
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def __len__(self):
        return len(self.paths)


class InferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        # src_path = '/root/data1/wangxiaolin/Unet-RegSeg/Data/ISeg/subject-4-T1.nii.gz'
        # sseg_path = '/root/data1/wangxiaolin/Unet-RegSeg/Data/ISeg/subject-4-label.nii.gz'
        # src_path = '/root/data1/wangxiaolin/Unet-RegSeg/Data/SKI10/9256759_vol.nii.gz'
        # sseg_path = '/root/data1/wangxiaolin/Unet-RegSeg/Data/SKI10/9256759_seg_4.nii.gz'
        # src_path = '/root/data1/wangxiaolin/Unet-RegSeg/Data/t2_zoomed/subject-4-T2.nii.gz'
        # sseg_path = '/root/data1/wangxiaolin/Unet-RegSeg/Data/t2_zoomed/subject-4-label.nii.gz'
        # src_path = '/user77/wangxiaolin/Unet-RegSeg/Data/atlas/atlas_vol.nii'
        # sseg_path = '/user77/wangxiaolin/Unet-RegSeg/Data/atlas/atlas_seg_14labels.nii'
        src_path = '/root/data1/wangxiaolin/Unet-RegSeg/Data/t2_zoomed/subject-4-T2.nii.gz'
        sseg_path = '/root/data1/wangxiaolin/Unet-RegSeg/Data/t2_zoomed/subject-4-label.nii.gz'
        tgt_path = self.paths[index]
        tseg_path = tgt_path.replace('T2', 'label')
        # tseg_path = tgt_path.replace('image', 'labels')
        # tseg_path = tgt_path.replace('vol', 'seg')


        tgt, src = load_volfile(tgt_path), load_volfile(src_path)
        tgt, src = normalize_image_intensity(tgt), normalize_image_intensity(src)

        tgt_seg, src_seg = load_volfile(tseg_path), load_volfile(sseg_path)  
        # tgt_seg = convert(tgt_seg)
        # src_seg = convert(src_seg)


        tgt = ndimage.zoom(tgt, (128 / 160, 128 / 192, 128 / 224), order=1)
        src = ndimage.zoom(src, (128 / 160, 128 / 192, 128 / 224), order=1)

        tgt_seg = ndimage.zoom(tgt_seg, (128 / 160, 128 / 192, 128 / 224), order=0)
        src_seg = ndimage.zoom(src_seg, (128 / 160, 128 / 192, 128 / 224), order=0)

        tgt, src = tgt[None, ...], src[None, ...]
        tgt_seg, src_seg= tgt_seg[None, ...], src_seg[None, ...]

        # tgt, tgt_seg = self.transforms([tgt, tgt_seg])
        # src, src_seg = self.transforms([src, src_seg])

        tgt = np.ascontiguousarray(tgt)# [Bsize,channelsHeight,,Width,Depth]
        src = np.ascontiguousarray(src)
        tgt_seg = np.ascontiguousarray(tgt_seg)  # [Bsize,channelsHeight,,Width,Depth]
        src_seg = np.ascontiguousarray(src_seg)

        tgt, src, tgt_seg, src_seg = torch.from_numpy(tgt), torch.from_numpy(src), torch.from_numpy(tgt_seg), torch.from_numpy(src_seg)
        # 返回顺序是source,target,source_seg,target_seg
        return src, tgt, src_seg, tgt_seg

    def __len__(self):
        return len(self.paths)

class VolDataset(Dataset):
    
    def __init__(self, train_vol_names):
        self.train_vol_names = train_vol_names
        '''class torchvision.transforms.ToTensor'''

    def __getitem__(self, i):
        Target_image, Source_image, _, _ = self.train_vol_names[i]
        image_0, image_1 = load_volfile(Target_image), load_volfile(Source_image)
        
        image_0 = ndimage.zoom(image_0, (160 / 160, 160 / 192, 160 / 224), order=1)
        image_1 = ndimage.zoom(image_1, (160 / 160, 160 / 192, 160 / 224), order=1)

        image_0 = torch.Tensor(image_0).float()
        image_0 = image_0.unsqueeze(0)
        # image_0 = image_0.permute(0, 2, 3, 1)
        image_1 = torch.Tensor(image_1).float()
        image_1 = image_1.unsqueeze(0)
        # image_1 = image_1.permute(0, 2, 3, 1)

        return image_0, image_1

    def __len__(self):
        return len(self.train_vol_names)

class PairsDataset(Dataset):
    
    def __init__(self, train_vol_names):
        self.train_vol_names = train_vol_names
        '''class torchvision.transforms.ToTensor'''

    def __getitem__(self, i):
        vol_path, seg_path = self.train_vol_names[i]
        vol, seg = load_volfile(vol_path), load_volfile(seg_path)
        
        vol = ndimage.zoom(vol, (160 / 160, 160 / 192, 160 / 224), order=1)
        seg = ndimage.zoom(seg, (160 / 160, 160 / 192, 160 / 224), order=0)

        vol = torch.Tensor(vol).float()
        vol = vol.unsqueeze(0)
        seg = torch.Tensor(seg).float()
        seg = seg.unsqueeze(0)

        return vol, seg

    def __len__(self):
        return len(self.train_vol_names)
