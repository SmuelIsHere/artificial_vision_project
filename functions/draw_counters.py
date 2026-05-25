FONT_SCALE_SMALL = 0.3
FONT_THICKNESS_SMALL = 1
FONT_SCALE_ID = 0.4
FONT_THICKNESS_ID = 1
BOX_PADDING = 6
import cv2
#funzione per disegnare i contatori per le persone totali viste nel video e i passaggi per le linee
def draw_counters(frame, total_people, line_counts, line_specs):
    counter_texts = ([f"Total people: {total_people}"] +
                     [f"Passages for line {line['id']}: {line_counts[line['id']-1]}" for _, line in enumerate(line_specs)])

    text_height = sum([cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, FONT_THICKNESS_SMALL)[0][1]
                       for text in counter_texts])
    box_height = text_height + BOX_PADDING * len(counter_texts)
    top_left = (2, 2)
    bottom_right = (10 + max([cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, FONT_THICKNESS_SMALL)[0][0]
                               for text in counter_texts]) + 10, top_left[1] + box_height)

    cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), -1)

    y_offset = top_left[1] + BOX_PADDING
    for text in counter_texts:
        cv2.putText(frame, text, (top_left[0] + 5, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, (0, 0, 0), FONT_THICKNESS_SMALL)
        y_offset += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, FONT_THICKNESS_SMALL)[0][1] + BOX_PADDING
    return frame