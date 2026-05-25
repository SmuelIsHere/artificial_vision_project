import cv2
def draw_line(frame, line, line_color=(255, 0, 0), line_thickness=3):  
    cv2.line(frame, (line["x1_start"], line["y1_start"]), (line["x1_finish"], line["y1_finish"]), line_color, line_thickness)
    if line["x1_start"] < line["x1_finish"]:
        label_position = (line["x1_start"], line["y1_start"] - 10)  
    else:
        label_position = (line["x1_finish"], line["y1_finish"] - 10) 
    line_id = line.get("id", "ID")  
    cv2.putText(frame, f"{line_id}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 1)  
    return frame