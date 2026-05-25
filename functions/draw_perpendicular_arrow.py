import cv2
import numpy as np
def draw_perpendicular_arrow(image, line, arrow_length=20, arrow_color=(255, 0, 0), thickness=3):

    x1, x2 = line["x1_start"], line["x1_finish"]
    y1, y2 = line["y1_start"], line["y1_finish"]
    
    # Calcola il vettore della linea
    if x1<x2 and y1<y2:
        dx, dy = x2 - x1, y2 - y1
    else:
        dx, dy = x1 - x2, y1 - y2

    # Trova il vettore perpendicolare
    perp_dx, perp_dy = -dy, dx

    # Normalizza il vettore perpendicolare
    length = np.sqrt(perp_dx ** 2 + perp_dy ** 2)
    perp_dx, perp_dy = (perp_dx / length) * arrow_length, (perp_dy / length) * arrow_length

    # Calcola il punto centrale della linea
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Calcola la posizione finale della freccia
    arrow_x, arrow_y = int(center_x + perp_dx), int(center_y + perp_dy)

    # Disegna la freccia
    cv2.arrowedLine(image, (center_x, center_y), (arrow_x, arrow_y), arrow_color, thickness, tipLength=0.3)
    return image
