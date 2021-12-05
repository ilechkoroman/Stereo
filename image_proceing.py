from utils import cost_map2edges
from graph import Graph
import numpy as np
import cv2


def read(path, resize=0.5):
    img = cv2.imread(path, 0)
    width = int(img.shape[1] * resize)
    height = int(img.shape[0] * resize)
    dim = (width, height)

    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def preprocess_best_path_idx(best_path: str) -> list:
    best_path_idx = best_path.split('|')[1:]
    best_path_idxs = list()

    for idx in best_path_idx:
        i, j = idx.split(',')
        best_path_idxs.append((int(i), int(j)))
    return best_path_idxs


class DynamicStereo:
    def __init__(self, f: float, cx: int, cy: int, doffs: float, baseline: float,
                 width: int, height: int, w_size: int, stride: int):
        self.f = f
        self.cx = cx
        self.cy = cy
        self.doffs = doffs
        self.baseline = baseline
        self.width = width
        self.height = height
        self.w_size = w_size
        self.stride = stride
        self.disparity_map = None

    def get_depth(self) -> np.array:
        return self.baseline * self.f / (self.disparity_map + self.doffs)

    def ssd(self, patch1: np.array, patch2: np.array) -> int:
        difference = patch1.ravel() - patch2.ravel()
        return difference@difference

    def get_w_row(self, img_left: np.array, img_right: np.array, i: int) -> np.array:
        return img_left[i: i + self.w_size, :], img_right[i: i + self.w_size, :]

    def get_error_map(self, left_w_row: np.array, right_w_row: np.array) -> np.array:
        error_map = np.zeros((self.width - self.stride, self.width - self.stride))
        for i in range(self.width - self.w_size + 1):
            left_patch = left_w_row[:, i: i + self.w_size]
            for j in range(self.width - self.w_size + 1):
                right_patch = right_w_row[:, j: j + self.w_size]
                error = self.ssd(left_patch, right_patch)
                error_map[i, j] = error
        return error_map

    def set_disparity_map(self, img_left: np.array, img_right: np.array) -> np.array:
        self.disparity_map = list()
        for i in range(self.height - self.w_size + 1):
            print(f'{i} from {self.height - self.w_size}')
            # make error_map from two rows
            left_w_row, right_w_row = self.get_w_row(img_left, img_right, i)
            error_map = self.get_error_map(left_w_row, right_w_row)
            print('Computed error map')

            # make edges, construct graph and find best path
            end_point = f'{error_map.shape[0] - 1},{error_map.shape[1]  - 1}'
            edges = cost_map2edges(error_map)
            print('Computed edges')
            g = Graph('-1,-1', end_point)
            g.add_weights(edges)
            g.find_all_dist()
            print('found best path')
            best_path = g.graph[end_point]
            print(best_path)

            best_path_idx = preprocess_best_path_idx(best_path['path'])

            self.disparity_map.append(self.make_disparity_row(best_path_idx))
        print('Computed disparity map')
        self.disparity_map = np.array(self.disparity_map)

    def make_disparity_row(self, best_path_idx):
        previous_idx = None
        disparity = [0]*self.width
        i = 0
        for idxs in best_path_idx:
            if previous_idx and (previous_idx[0] == idxs[0] or previous_idx[1] == idxs[1]):
                # occlusion occurred we need to go further
                continue
            disparity[i] = idxs[1] - i
            i += 1
            previous_idx = idxs
        return np.array(disparity)
