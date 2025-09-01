import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label):

        w,h = image.size[0], image.size[1]
        padw = self.size[0]-w if w<self.size[0] else 0
        padh = self.size[0]-h if h<self.size[0] else 0
        if padw!=0 or padh!=0:
            image = F.pad(image, (0,0,padw,padh), padding_mode='reflect')
            label = F.pad(label, (0,0,padw,padh), padding_mode='reflect')

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)

class PairCenterCrop(transforms.CenterCrop):
    def __call__(self, image, lable):

        image = F.center_crop(image, (self.size[0], self.size[0]))
        lable = F.center_crop(lable, (self.size[0], self.size[0]))

        return image, lable


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
    def __call__(self, image, label):
        """
        Args:
            image (PIL Image): Image to be resized.
            label (PIL Image): Label to be resized.

        Returns:
            tuple: Resized image and label.
        """
        h, w = self.size

        # width, height = image.size

        size = (w, h)

        # if width > height:
        #     size = (w, h)
        # else:
        #     size = (h, w)

        return F.resize(image, size), F.resize(label, size)



class PairCenterCrop(transforms.CenterCrop):
    def __call__(self, image, label):

        image = F.center_crop(image, (self.size[0], self.size[0]))
        label = F.center_crop(label, (self.size[0], self.size[0]))

        return image, label


class PairRandomRotation(transforms.RandomRotation):
    def __call__(self, image, label):
        if random.random() < 0.5:
            angle = self.get_params(self.degrees)
            return F.rotate(image, angle), F.rotate(label, angle)
        return image, label