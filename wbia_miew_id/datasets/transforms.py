import numpy
import cv2
import glob
import albumentations
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch.transforms import ToTensorV2


def triangle(img, p):
    xx = numpy.random.rand(1)[0]
    if xx > p:
        h, w, _= img.shape
        limitw = int(w * 0.3)
        limith = int(h * 0.25)
        desc = 0
        step = limitw / limith
        for i in range(limith):
            img[i][:limitw - int(step * i)] = (255, 255, 255)
    return img

class Triangle(ImageOnlyTransform):
    def __init__(self, p):
        super(Triangle, self).__init__(p)
        self.p = p
    def apply(self, img , **params):
        return triangle(img , self.p)

def get_train_transforms(config):
    return albumentations.Compose(
        [   Triangle(p = 0.5),
            albumentations.Resize(config.data.image_size[0],config.data.image_size[1],always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            
            #albumentations.VerticalFlip(p=0.5),
            #albumentations.ImageCompression (quality_lower=50, quality_upper=100, p = 0.5),
            albumentations.OneOf([
            albumentations.Sharpen(p=0.3),
            albumentations.ToGray(p=0.3),
            albumentations.CLAHE(p=0.3),
            ], p=0.5),
            #albumentations.Rotate(limit=30, p=0.8),
            #albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.7),
            #albumentations.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.3),
            albumentations.ShiftScaleRotate(
               shift_limit=0.25, scale_limit=0.2, rotate_limit=4,p = 0.5
            ),
            albumentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            albumentations.Normalize(),
            ToTensorV2(),
        ]
    )

def get_valid_transforms(config):

    return albumentations.Compose(
        [
            albumentations.Resize(config.data.image_size[0],config.data.image_size[1],always_apply=True),
            albumentations.Normalize(),
        ToTensorV2(p=1.0)
        ]
    )



def get_test_transforms(config):

    return albumentations.Compose(
        [
            albumentations.Resize(config.data.image_size[0],config.data.image_size[1],always_apply=True),
            albumentations.Normalize(),
        ToTensorV2(p=1.0)
        ]
    )