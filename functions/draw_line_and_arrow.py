from functions.draw_line import draw_line
from functions.draw_perpendicular_arrow import draw_perpendicular_arrow

def draw_line_and_arrow(frame, line, line_color=(255, 0, 0)):
    frame = draw_line(frame, line, line_color)
    frame = draw_perpendicular_arrow(frame, line)
    return frame