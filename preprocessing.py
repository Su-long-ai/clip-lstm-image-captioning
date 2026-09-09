from __future__ import annotations

from PIL import ImageEnhance


class Sharpen:
    def __init__(self, factor: float = 1.5):
        self.factor = float(factor)

    def __call__(self, image):
        return ImageEnhance.Sharpness(image).enhance(self.factor)
