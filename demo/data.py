import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from multipledispatch import dispatch 
from PIL.Image import Image

class PairToTensor(transforms.ToTensor):

    @dispatch(Image, Image)
    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)
    
    @dispatch(Image, type(None))
    def __call__(self, pic, _):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic)


class PairResize(transforms.Resize):

    def __init__(self, size, interpolation=transforms.InterpolationMode.BILINEAR):
        super().__init__(size, interpolation)
        
        if isinstance(self.size, tuple):
            w, h = self.size
            self.size = (h, w)

    @dispatch(Image, Image)
    def __call__(self, image, label):
        """
        Args:
            image (PIL Image): Image to be resized.
            label (PIL Image): Label to be resized.

        Returns:
            tuple: Resized image and label.
        """

        return super().__call__(image), super().__call__(label)
    
    @dispatch(Image, type(None))
    def __call__(self, image, _):
        """
        Args:
            image (PIL Image): Image to be resized.
            label (PIL Image): Label to be resized.

        Returns:
            tuple: Resized image and label.
        """

        return super().__call__(image)