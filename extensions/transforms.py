import numbers
import numpy as np

import torch

from torchvision.transforms import functional as F
from torchvision.transforms import functional_pil as F_pil, functional_tensor as F_t


class GaussianBlur(object):

    def __init__(self, sigma):
        self.sigma = sigma

        if isinstance(sigma, numbers.Number):
            self.min_sigma = sigma
            self.max_sigma = sigma
        elif isinstance(sigma, list):
            if len(sigma) != 2:
                raise Exception("`sigma` should be a number or a list of two numbers")
            if sigma[1] < sigma[0]:
                raise Exception(
                    "radius[0] should be <= radius[1]")
            self.min_sigma = sigma[0]
            self.max_sigma = sigma[1]
        else:
            raise Exception("`sigma` should be a number or a list of two numbers")

    def __call__(self, image):
        sigma = np.random.uniform(self.min_sigma, self.max_sigma)
        sigma = max(0.0, sigma)
        ksize = int(sigma+0.5) * 8 + 1
        return F.gaussian_blur(image, ksize)

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


class RandomAffine(object):

    def __init__(self, mean=0.0, stdv=0.1):
        self.mean = mean
        self.stdv = stdv

    def __call__(self, image):
        a = 1 + np.random.normal(loc=self.mean, scale=self.stdv)
        b = np.random.normal(loc=self.mean, scale=self.stdv)
        c = np.random.normal(loc=self.mean, scale=self.stdv)
        d = 1 + np.random.normal(loc=self.mean, scale=self.stdv)
        if F_pil._is_pil_image(image):
            return F_pil.affine(image, [a, b, 0.0, c, d, 0.0])
        else:
            return F_t.affine(image, [a, b, 0.0, c, d, 0.0])

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, stdv={1})'.format(self.mean, self.stdv)

