import torch
from torchvision import transforms
from torchvision.transforms import functional as F


class Resize(object):
    """Resize the input image to the given size.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).
            In torchscript mode padding as single int is not supported, use a tuple or
            list of length 1: ``[size, ]``.
    """

    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        self.size = size

    def __call__(self, sample):
        t = transforms.Resize(self.size)
        return {'original_image': t(sample['original_image']),
                'downsampled_image': t(sample['downsampled_image']),
                'label': sample['label']}


class RandomHorizontalFlip(object):
    """Horizontally flip the given image randomly with a given probability.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            return {'original_image': F.hflip(sample['original_image']),
                    'downsampled_image': F.hflip(sample['downsampled_image']),
                    'label': sample['label']}
        return sample


class RandomCrop(object):
    """Crop the given image at a random location.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.
            In torchscript mode padding as single int is not supported, use a tuple or
            list of length 1: ``[padding, ]``.
    """

    def __init__(self, size, padding=None):
        self.size = size
        self.padding = padding

    def __call__(self, sample):
        original_image = sample['original_image']
        downsampled_image = sample['downsampled_image']

        if self.padding is not None:
            original_image = F.pad(original_image, self.padding)
            downsampled_image = F.pad(downsampled_image, self.padding)

        i, j, h, w = transforms.RandomCrop.get_params(original_image, self.size)

        return {'original_image': F.crop(original_image, i, j, h, w),
                'downsampled_image': F.crop(downsampled_image, i, j, h, w),
                'label': sample['label']}


class ToTensor:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, sample):
        return {'original_image': F.to_tensor(sample['original_image']),
                'downsampled_image': F.to_tensor(sample['downsampled_image']),
                'label': sample['label']}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        t = transforms.Normalize(self.mean, self.std, self.inplace)
        return {'original_image': t(sample['original_image']),
                'downsampled_image': t(sample['downsampled_image']),
                'label': sample['label']}
