import numpy as np
import cv2

from PIL import Image, ImageDraw, ImageFont

import Helpers.Geometry as Geo


class DrawObj:
    def __init__(self, image_in, draw_in):
        self.image = image_in
        self.draw = draw_in

    def to_array(self):
        return np.array(self.image)

class Draw:

    @staticmethod
    def begin(array_in):
        image = Image.fromarray(array_in)
        draw = ImageDraw.Draw(image)
        return DrawObj(image, draw)

    @staticmethod
    def end(draw_in:DrawObj):
        return draw_in.to_array()

    @staticmethod
    def points(draw_in, points, color=(45, 197, 252)):
        draw:ImageDraw.Draw = draw_in.draw
        draw.point(points, fill=color)

    @staticmethod
    def lines(draw_in, lines, line_color=(45, 197, 252), thickness=3):
        draw = draw_in.draw

        if isinstance(lines, np.ndarray):
            lines = [tuple(map(tuple, line)) for line in lines.tolist()]

        for line in lines:
            draw.line(line, fill=line_color, width=thickness)

    @staticmethod
    def rectangle(draw_in, rect, line_color=(45, 197, 252), thickness=3):
        draw = draw_in.draw
        draw.rectangle(rect, fill=None, outline=line_color, width=thickness)

    @staticmethod
    def rectangle_grid(draw_in, rect, grid_width, line_color=(255, 255, 255), thickness=1):
        if grid_width == 0:
            return None
            
        origin, top_right = rect
        width = Geo.Rectangle.width(rect)
        height = Geo.Rectangle.height(rect)
        cell_width = width / grid_width

        grid_height = int(round(height / cell_width, 0))

        x_min, y_min = origin
        x_max, _ = top_right
        y_max = y_min - (grid_height * cell_width)

        vert_lines = []
        for x in range(0, grid_width+1):
            each_x = x_min + (x * cell_width)
            vert_lines += [((each_x, y_min), (each_x, y_max))]

        hori_lines = []
        for y in range(0, grid_height+1):
            each_y = y_min - (y * cell_width)
            hori_lines += [((x_min, each_y), (x_max, each_y))]

        Draw.lines(draw_in, vert_lines+hori_lines, line_color, thickness)  

    @staticmethod
    def text(draw_in, text, position, color=(0, 0, 0), font_size=15):
        draw = draw_in.draw
        font = ImageFont.truetype("resources/fonts/Roboto-Regular.ttf", font_size)
        draw.text(position, text, fill=color, font=font)
