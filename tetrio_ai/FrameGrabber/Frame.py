import numpy as np
import cv2

import Helpers.Geometry as Geo


class cmap:
    hsv  = cv2.COLOR_BGR2HSV
    hls  = cv2.COLOR_BGR2HLS
    gray = cv2.COLOR_BGR2GRAY

class ArrayComponents(np.ndarray):
    def __new__(cls, input_array, components):
        obj = np.asarray(input_array).view(cls)
        for idx, c in enumerate(components):
            obj.__dict__[c] = input_array[:, :, idx]

        return obj

class Frame(np.ndarray):

    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)

    def __array_finalize__(self, obj):
        if obj is None: return

        self.source = self
        self.conversions = {
            cmap.hsv : None,
            cmap.hls : None,
            cmap.gray: None,
        }

    @staticmethod
    def translate_roi_from_cartesian(frame, roi):
        origin, top_right = list(map(list, roi))

        frame_height = frame.shape[0]

        origin[Geo.Point.y] = frame_height - origin[Geo.Point.y]
        top_right[Geo.Point.y] = frame_height - top_right[Geo.Point.y]

        return (tuple(origin), tuple(top_right))

    @staticmethod
    def change_nchannels(frame, nchannels):
        init_height = frame.shape[0]
        init_width = frame.shape[1]

        new_frame = frame.copy()
        new_shape = (init_height, init_width, nchannels)
        return new_frame.ravel().repeat(nchannels).reshape(new_shape)

    @staticmethod
    def nchannels(frame):
        if frame.ndim < 3:
            return 1
        return frame.shape[2]

    @staticmethod
    def hstack(frames, padding_color=(128, 128, 128)):
        channel_count = Frame.nchannels(frames[0])
        max_height = max([frame.shape[0] for frame in frames])
        output_im = np.zeros((max_height, 0, channel_count), dtype=np.int8)

        for frame in frames:
            each_height = frame.shape[0]
            if each_height != max_height:
                height_delta = max_height - each_height
                top = height_delta // 2
                bottom = height_delta - top
                frame = Frame.add_padding(frame, top=top, bottom=bottom, padding_color=padding_color)

            output_im = np.hstack((output_im, frame)) 

        return output_im

    @staticmethod
    def add_padding(frame, top=0, right=0, bottom=0, left=0, padding_color=(128, 128, 128)):
        color_channels_count = Frame.nchannels(frame)
        
        if color_channels_count == 1:
            padding_color = list(padding_color)[0]

        if left != 0:
            height = frame.shape[0]
            h_bar_shape = (height, left, color_channels_count)
            if color_channels_count == 1:
                h_bar_shape = (height, left)

            h_bar = np.full(h_bar_shape, padding_color, np.uint8)
            frame = np.hstack((h_bar, frame))

        if right != 0:
            height = frame.shape[0]
            h_bar_shape = (height, right, color_channels_count)
            if color_channels_count == 1:
                h_bar_shape = (height, right)

            h_bar = np.full(h_bar_shape, padding_color, np.uint8)
            frame = np.hstack((frame, h_bar))

        if top != 0:
            width = frame.shape[1]
            v_bar_shape = (top, width, color_channels_count)
            if color_channels_count == 1:
                v_bar_shape = (top, width)

            v_bar = np.full(v_bar_shape, padding_color, np.uint8)
            frame = np.vstack((v_bar, frame))

        if bottom != 0:
            width = frame.shape[1]
            v_bar_shape = (bottom, width, color_channels_count)
            if color_channels_count == 1:
                v_bar_shape = (bottom, width)

            v_bar = np.full(v_bar_shape, padding_color, np.uint8)
            frame = np.vstack((frame, v_bar))

        return frame

    @staticmethod
    def compare_shape(frame_1, frame_2):
        if frame_1 is None or frame_2 is None:
            return False
        
        h1, w1 = frame_1.shape
        h2, w2 = frame_2.shape
        c1 = h1 == h2
        c2 = w1 == w2
        return all([c1, c2])

    @staticmethod
    def crop(frame, roi):
        nframe = frame.copy()
        y_min = roi[0][Geo.Point.y]
        y_max = roi[1][Geo.Point.y]

        x_min = roi[0][Geo.Point.x]
        x_max = roi[1][Geo.Point.x]
        return nframe[y_max:y_min+1, x_min:x_max+1]

    @staticmethod
    def create_mask(frame, rois:list):
        h = frame.shape[0]
        w = frame.shape[1]
        mask = np.zeros((h, w), dtype=np.uint8)

        for roi in rois:
            origin_x, origin_y = roi[0]
            top_x, top_y = roi[1]

            y_lower = origin_y+1
            x_min = origin_x

            y_upper = top_y
            x_max = top_x+1

            mask[y_upper:y_lower, x_min:x_max] = 1

        return mask

    @staticmethod
    def resize(frame, target_h=-1):
        if frame is None:
            return None
        
        frame = frame.astype(np.uint8)
        
        if target_h != -1:
            ih, iw = (frame.shape[0], frame.shape[1])
            h = target_h
            w = h * iw // ih
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        
        return frame

    def convert_color(self, cmap_code):
        if self.conversions[cmap_code] is None:
            self.conversions[cmap_code] = cv2.cvtColor(self.source, cmap_code)
        return self.conversions[cmap_code]

    @property
    def height(self):
        return self.source.shape[0]

    @property
    def width(self):
        return self.source.shape[1]

    @property
    def hsv(self):
        return ArrayComponents(self.convert_color(cmap.hsv), "hsv")

    @property
    def hls(self):
        return ArrayComponents(self.convert_color(cmap.hls), "hls")

    @property
    def gray(self):
        return self.convert_color(cmap.gray)
