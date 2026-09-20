import numpy as np
import cv2
import matplotlib.patches as patches
from warnings import warn


def _2dim_requirements(element):
    if len(element) != 2 or not all(isinstance(item, int) for item in element):
        raise Exception('ROI: element must be an iterable with 2 integers.')


def _rect_requirements(rect):
    rect = rect[:]
    if not isinstance(rect, tuple) or len(rect) != 4 or not all(isinstance(item, int) for item in rect):
        raise Exception('ROI: rect must be a ROI or a tuple with 4 integers')

class ROI:
    def __init__(self, region, rect=None, select=False):
        self._region = region if isinstance(region, tuple) else region.shape
        _2dim_requirements(self._region)

        height, width = self._region
        self._rect = (0, 0, width, height)

        if rect is not None:
            _rect_requirements(rect)
            self._rect = tuple(rect[:])

        elif select:
            cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
            self._rect = cv2.selectROI('Select ROI', region)
            cv2.destroyWindow('Select ROI')

    def __getitem__(self, item):
        return self._rect[item]

    def __add__(self, vector):
        _2dim_requirements(vector)
        rect = (self[0] + vector[0], self[1] + vector[1], self[2], self[3])
        return ROI(region=self._region, rect=rect)

    def __sub__(self, vector):
        _2dim_requirements(vector)
        rect = (self[0] - vector[0], self[1] - vector[1], self[2], self[3])
        return ROI(region=self._region, rect=rect)

    def __mul__(self, factor):
        rect = (self[0], self[1], self[2] * factor, self[3] * factor)
        return ROI(region=self._region, rect=rect).recenter(self.center)

    def __floordiv__(self, factor): 
        rect = (self[0], self[1], self[2] // factor, self[3] // factor)
        return ROI(region=self._region, rect=rect).recenter(self.center)
    
    def __str__(self):
        return f'ROI: {self.rect}, center: {self.center}, shape: {self.shape}'

    @property
    def rect(self):
        return self._rect

    @property
    def shape(self): 
        return (self[3], self[2])

    @property
    def center(self):
        shape = self.shape
        return (self[0] + shape[1] // 2, self[1] + shape[0] // 2)
    
    @property
    def corner(self):
        return (self[0], self[1])

    @property
    def limits(self):
        return (self[0], self[0] + self[2], self[1], self[1] + self[3])

    @property
    def xi(self):
        return self.limits[0]

    @property
    def xf(self):
        return self.limits[1]

    @property
    def yi(self):
        return self.limits[2]

    @property
    def yf(self):
        return self.limits[3]


    def reset(self):
        height, width = self._region
        rect = (0, 0, width, height)
        return ROI(region=self._region, rect=rect)

    def local_to_absolute(self, local):
        _2dim_requirements(local)
        return (local[0] + self[0], local[1] + self[1])

    def absolute_to_local(self, absolute):
        _2dim_requirements(absolute)
        return (absolute[0] - self[0], absolute[1] - self[1])

    def recenter(self, new_center):
        _2dim_requirements(new_center)
        return self - self.center + new_center

    def reshape(self, new_shape): 
        _2dim_requirements(new_shape)
        return ROI(region=self._region, rect=(self[0], self[1], new_shape[1], new_shape[0])).recenter(self.center)

    def recorner(self, new_corner): 
        _2dim_requirements(new_corner)
        return self - self.corner + new_corner


    def scale(self, factor): 
        return self * factor

    def rotate(self, center, angle): 
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        M = rot_mat[:,:2]
        v = rot_mat[:,2]
        new_center = (M @ np.array(self.center) + v).astype('int').tolist()
        return self.recenter(new_center)

    def square(self):
        center = self.center
        rect = (self[0], self[1]) + (min(self.shape),)*2
        return ROI(region=self._region, rect=rect).recenter(center)

    def apply_bounds(self):
        image = np.zeros(self._region)
        x, y, w, h = self._rect
        new_h, new_w = image[max(0, y):y+h, max(0, x):x+w].shape
        if new_h == 0 or new_w == 0: 
            warn('ROI: roi is out of image bounds and has been reset')
            return self.reset()
        new_x, new_y = max(0, self[0]), max(0, self[1])
        return ROI(region=self._region, rect = (new_x, new_y, new_w, new_h))

    @property
    def patch(self):
        rect = patches.Rectangle(self.corner, self[2], self[3], linewidth=1, edgecolor='r', facecolor='none')
        return rect

    
if __name__ == '__main__' : 
    image = np.zeros((100, 150))
    roi1 = ROI(image)
    roi2 = ( roi1 + (3, 4) )*2
    roi3 = roi2.recenter((0,0))

    print(roi1)
    print(roi2)
    print(roi3)