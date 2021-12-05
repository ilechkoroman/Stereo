import numpy as np
from easydict import EasyDict


def cost_map2edges(error_map: np.array) -> list:
    edges = list()
    w, h = error_map.shape

    for i in range(w):
        if i == 0:
            edges.append(('-1,-1', '0,0', error_map[0, 0]))
        for j in range(h):
            edges.extend(get_neighbours(i, j, error_map, w, h))
    return edges


def get_neighbours(i: int, j: int, error_map: np.array, w: int, h: int) -> list:
    neighbours = list()
    if i != 0:
        # can add neighbours from top
        edge = (f'{i},{j}', f'{i - 1},{j}', error_map[i - 1, j])
        neighbours.append(edge)
        if j != h - 1:
            # can add neighbours from top right
            edge = (f'{i},{j}', f'{i - 1},{j + 1}', error_map[i - 1, j + 1])
            neighbours.append(edge)
    if i != w - 1:
        # can add neighbours from bot
        edge = (f'{i},{j}', f'{i + 1},{j}', error_map[i + 1, j])
        neighbours.append(edge)
        if j != h - 1:
            # can add neighbours from bot right
            edge = (f'{i},{j}', f'{i + 1},{j + 1}', error_map[i + 1, j + 1])
            neighbours.append(edge)
    if j != h - 1:
        # can add neighbours from right
        edge = (f'{i},{j}', f'{i},{j + 1}', error_map[i, j + 1])
        neighbours.append(edge)
    return neighbours


def get_params(calib_param: str, resize: float) -> EasyDict:
    calib_params = EasyDict()
    lines = calib_param.split('\n')
    for line in lines:
        if 'cam' in line:
            params = line.split('[')[1][:-1].split(' ')
            params = [param.replace(';', '') for param in params]
            params = [float(param) for param in params]
            calib_params.f = params[0]*resize

            calib_params.cx = params[2]*resize
            calib_params.cy = params[5]*resize
        else:
            res = line.split('=')
            if '' in res:
                continue
            param_name, value = res[0], float(res[1])
            if param_name == 'width' or param_name == 'height':
                calib_params[param_name] = int(value*resize)
            else:
                calib_params[param_name] = value * resize
    return calib_params
