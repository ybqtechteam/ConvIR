import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, w, h = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCenterCrop(transforms.CenterCrop):

    def __init__(self, size, fill=128, padding_mode='constant'):
        super().__init__(size)
        self.fill = fill
        self.padding_mode = padding_mode
        self._is_padded = False

        if isinstance(self.size, tuple):
            w, h = self.size
            self.size = (h, w)
    

    def __call__(self, image, label):
        self._is_padded = False
        
        image = self._pad(image)
        label = self._pad(label)
        
        if self._is_padded:
            return image, label
        
        return super().__call__(image), super().__call__(label)
    
    def _pad(self, img):
        img_w, img_h = img.size
        crop_h, crop_w = self.size

        if crop_w > img_w or crop_h > img_h:
            padding_ltrb = [
                (crop_w - img_w) // 2 if crop_w > img_w else 0,  # left
                (crop_h - img_h) // 2 if crop_h > img_h else 0,  # top
                (crop_w - img_w + 1) // 2 if crop_w > img_w else 0,  # right
                (crop_h - img_h + 1) // 2 if crop_h > img_h else 0   # bottom
            ]
            img = F.pad(img, padding_ltrb, self.fill, self.padding_mode)
            self._is_padded = True
            img_w, img_h = img.size

            if crop_w == img_w and crop_h == img_h:
                return img
        
        crop_t = int(round((img_h - crop_h) / 2.0))
        crop_l = int(round((img_w - crop_w) / 2.0))
        return F.crop(img, crop_t, crop_l, crop_h, crop_w)

class PairCompose(transforms.Compose):
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label

class PairRandomVerticalFlip(transforms.RandomVerticalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.vflip(img), F.vflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)
    

class PairResize(transforms.Resize):

    def __init__(self, size, interpolation=transforms.InterpolationMode.BILINEAR):
        super().__init__(size, interpolation)
        
        if isinstance(self.size, tuple):
            w, h = self.size
            self.size = (h, w)

    def __call__(self, image, label):
        """
        Args:
            image (PIL Image): Image to be resized.
            label (PIL Image): Label to be resized.

        Returns:
            tuple: Resized image and label.
        """

        return super().__call__(image), super().__call__(label)


class PairRandomRotation(transforms.RandomRotation):
    def __call__(self, image, label):
        if random.random() < 0.5:
            angle = self.get_params(self.degrees)
            return F.rotate(image, angle), F.rotate(label, angle)
        return image, label