import cv2
from utils import get_params
from image_proceing import DynamicStereo, read

resize_value = 0.03
left_img = read('data/Adirondack-perfect/im0.png', resize_value)
right_img = read('data/Adirondack-perfect/im1.png', resize_value)


with open('data/Adirondack-perfect/calib.txt') as f:
    calib_param = f.read()


calib_param = get_params(calib_param, resize_value)
print(calib_param)

alg = DynamicStereo(calib_param.f,
                    calib_param.cx,
                    calib_param.cy,
                    calib_param.doffs,
                    calib_param.baseline,
                    calib_param.width,
                    calib_param.height,
                    45,
                    1)
alg.set_disparity_map(left_img, right_img)
print(alg.disparity_map)
print('computing depth')
print(alg.get_depth())
cv2.imshow('depth', alg.get_depth())
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
