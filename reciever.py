import cv2  # type: ignore
import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from sensor_msgs.msg import CompressedImage  # type: ignore
from rclpy.qos import QoSProfile  # type: ignore
from queue import Queue, Full, Empty
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
__TOPIC__ = '/panorama/image/compressed'
__node__ = None
__subscriber__ = None
xmap = np.loadtxt('xmap.pgm').astype(np.float32)
ymap = np.loadtxt('ymap.pgm').astype(np.float32)
__WIDTH__ = 512
__HEIGHT__ = 256
rot = R.from_euler('xyz', [0, 270, 0], degrees=True).as_matrix()


def init():
    global __node__
    if __node__ is None:
        rclpy.init()
        __node__ = rclpy.create_node("usb_webcam_display")


def subscribe(callback):
    global __subscriber__, __node__
    if __node__ is None:
        return
    if __subscriber__ is None:
        __subscriber__ = __node__.create_subscription(CompressedImage, __TOPIC__, callback, 1)


def spherical_project(XYZ):
    lat = np.arcsin(XYZ[..., 2])
    lon = np.arctan2(XYZ[..., 1], XYZ[..., 0])

    x = lon / np.pi
    y = lat / np.pi * 2

    xy = np.stack((x, y), axis=-1)
    return xy


def spherical_project_inverse(xy, r: float = 1.0):
    lon = np.pi * xy[..., 0]
    lat = np.pi * xy[..., 1] / 2

    XYZ = np.stack(
        (
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ),
        axis=-1,
    )
    return XYZ


def remap_thubnail(img):
    res = cv2.remap(img, xmap.astype(np.float32), ymap.astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    x, y = np.meshgrid(np.arange(__WIDTH__), np.arange(__HEIGHT__))
    x = x / __WIDTH__ * 2 - 1
    y = y / __HEIGHT__ * 2 - 1
    xy = np.stack((x, y), axis=-1)
    xyz = spherical_project_inverse(xy)
    xyz = xyz @ rot.T
    xy = spherical_project(xyz)
    x = xy[..., 0]
    y = xy[..., 1]
    xmap2 = (x + 1) / 2 * __WIDTH__
    ymap2 = (y + 1) / 2 * __HEIGHT__
    res = cv2.remap(res, xmap2.astype(np.float32), ymap2.astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    return res
    
    
def main():
    init()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    
    def callback(msg):
        print('got image')
        jpg = msg.data
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = remap_thubnail(img)
        cv2.imshow('image', img)
        cv2.imwrite('test.jpg', img)
        cv2.waitKey(1)
    subscribe(callback)
    rclpy.spin(__node__)


if __name__ == "__main__":
    main()
