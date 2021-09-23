import math

class Point:
    x = 0
    y = 1

    @staticmethod
    def x_distance(p1, p2):
        x1 = p1[Point.x]
        x2 = p2[Point.x]
        return abs(x1 - x2)
        
    @staticmethod
    def y_distance(p1, p2):
        y1 = p1[Point.y]
        y2 = p2[Point.y]
        return abs(y1 - y2)

    @staticmethod
    def distance(p1, p2):
        x_dist = Point.x_distance(p1, p2)
        y_dist = Point.y_distance(p1, p2)
        return math.sqrt(x_dist**2 + y_dist**2)

    @staticmethod
    def slope(p1, p2):
        return (p2[Point.y] - p1[Point.y]) / (p2[Point.x] - p1[Point.x])

    @staticmethod
    def y_in_range(pt_to_check, y_lower, y_upper):
        y_check = pt_to_check[Point.y]
        return y_lower <= y_check <= y_upper

    @staticmethod
    def does_y_overlap(p1, p2, pt_to_check):
        y_min = min(p1[Point.y], p2[Point.y])
        y_max = max(p1[Point.y], p2[Point.y])
        y_check = pt_to_check[Point.y]
        return y_min <= y_check <= y_max

    @staticmethod
    def x_in_range(pt_to_check, x_lower, x_upper):
        x_check = pt_to_check[Point.x]
        return x_lower <= x_check <= x_upper

    @staticmethod
    def does_x_overlap(p1, p2, pt_to_check):
        x_min = min(p1[Point.x], p2[Point.x])
        x_max = max(p1[Point.x], p2[Point.x])
        x_check = pt_to_check[Point.x]
        return x_min <= x_check <= x_max


class Line:
    @staticmethod
    def length(line):
        p1, p2 = line
        return Point.distance(p1, p2)

    @staticmethod
    def does_y_overlap(first_line, second_line):
        p1, p2 = first_line
        p3, p4 = second_line

        c1 = Point.does_y_overlap(p1, p2, p3)
        c2 = Point.does_y_overlap(p1, p2, p4)
        c3 = Point.does_y_overlap(p3, p4, p1)
        c4 = Point.does_y_overlap(p3, p4, p2)

        return any([c1, c2, c3, c4])

    @staticmethod
    def does_x_overlap(first_line, second_line):
        p1, p2 = first_line
        p3, p4 = second_line

        c1 = Point.does_x_overlap(p1, p2, p3)
        c2 = Point.does_x_overlap(p1, p2, p4)
        c3 = Point.does_x_overlap(p3, p4, p1)
        c4 = Point.does_x_overlap(p3, p4, p2)

        return any([c1, c2, c3, c4])

    @staticmethod
    def y_distance(first_line, second_line):
        if Line.does_y_overlap(first_line, second_line):
            return 0

        p1, p2 = first_line
        p3, p4 = second_line

        d0 = Point.y_distance(p1, p3)
        d1 = Point.y_distance(p1, p4)
        d2 = Point.y_distance(p2, p3)
        d3 = Point.y_distance(p2, p4)

        return min(d0, d1, d2, d3)

    @staticmethod
    def x_distance(first_line, second_line):
        if Line.does_x_overlap(first_line, second_line):
            return 0

        p1, p2 = first_line
        p3, p4 = second_line

        d0 = Point.x_distance(p1, p3)
        d1 = Point.x_distance(p1, p4)
        d2 = Point.x_distance(p2, p3)
        d3 = Point.x_distance(p2, p4)

        return min(d0, d1, d2, d3)

class Rectangle:
    @staticmethod
    def roi_from_points(points_array):
        if len(points_array) == 0:
            return None
        
        xs = [pt[Point.x] for pt in points_array]
        ys = [pt[Point.y] for pt in points_array]

        x_min = min(xs)
        x_max = max(xs)
        y_lower = max(ys)
        y_upper = min(ys)

        origin = (x_min, y_lower)
        top_right = (x_max, y_upper)
        return (origin, top_right)

    @staticmethod
    def width(rect):
        origin, top_right = rect
        return Point.x_distance(origin, top_right)+1

    @staticmethod
    def height(rect):
        origin, top_right = rect
        return Point.y_distance(origin, top_right)+1
