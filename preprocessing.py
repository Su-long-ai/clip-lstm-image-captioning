from PIL import ImageFilter, ImageEnhance
import torchvision.transforms.functional as TF

class Sharpen(object):
    def __init__(self, factor=1.5):
        self.factor = factor
        
    def __call__(self, img):
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(self.factor)
        return img