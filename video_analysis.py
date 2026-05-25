import cv2
from ultralytics import YOLO
from classes.AttributeTracker import AttributeTracker
from functions.update_lines_crossed import person_lines_crossed

# Configurazione
THRESH_HAT = 0.65
THRESH_BAG = 0.4
THRESH_GEN = 0.55
SCALE_FACTOR = 1.7
FRAME_SKIP = 20
VIDEO_FILE = "video.mp4"
YOLO_MODEL_PATH = "yolo11s.pt"
TRACKER_CONFIG = "yaml/botsort.yaml"
LINE_SPECS_LOADER = "functions.load_line_specs"
FRAME_WAIT_BEFORE_PAR = 10
model = YOLO(YOLO_MODEL_PATH)

# Setup del video
video_path = f"videos/{VIDEO_FILE}"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Yolo preferisce risoluzioni multiplo di 32
from functions.scale_to_multiple_of_32 import scale_to_multiple_of_32
width_track = scale_to_multiple_of_32(frame_width, SCALE_FACTOR)
height_track = scale_to_multiple_of_32(frame_height, SCALE_FACTOR)


from functions.load_line_specs import load_line_specs
from functions.get_track_ids_and_y2 import get_track_ids_and_y2

line_specs = load_line_specs()

from functions.draw_counters import draw_counters
from functions.process_tracks import process_tracks

# Loop principale
frame_count = 0
people_counts = [0] * len(line_specs)
total_people_set = set()
people_info_dict = {}
results = None
track_info1 = None
ids_classification_story = AttributeTracker(FRAME_WAIT_BEFORE_PAR) 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Da passare al classificatore
    if frame is not None:
        resized_frame_no_boxes = cv2.resize(frame, (width_track, height_track), interpolation=cv2.INTER_LINEAR)

    if results is not None:
        track_info1 = get_track_ids_and_y2(results)

    if frame_count % FRAME_SKIP == 0:
        resized_frame = cv2.resize(frame, (width_track, height_track), interpolation=cv2.INTER_LINEAR)

        results = model.track(source=resized_frame, tracker=TRACKER_CONFIG,
                              conf=0.25, iou=0.6, classes=[0], persist=True, imgsz=(height_track, width_track))
        
        track_info2 = get_track_ids_and_y2(results)
        resized_frame,ids_classification_story = process_tracks(results, track_info1, track_info2, resized_frame,
                                        resized_frame_no_boxes, total_people_set,
                                          line_specs, people_counts, ids_classification_story)
        
        resized_frame = draw_counters(resized_frame, len(total_people_set), people_counts, line_specs)

        cv2.imshow("Tracking Online", resized_frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
prob = {}
ids_to_remove = []
for id in ids_classification_story.data.keys():
    if ids_classification_story.get_probabilities_count(id) < 3:
        ids_to_remove.append(id)
for ids in ids_to_remove:
    ids_classification_story.clear_data(ids)        
with open("results.txt", "w") as f:
    f.write('{"people": [\n')
    for id in ids_classification_story.data.keys():
        f.write("\t{\n")
        f.write(f'\t   "id": {int(id)},\n')
        prob = ids_classification_story.get_average_probabilities(id)
        if prob["gender_avg"]>THRESH_GEN:
            f.write(f'\t   "gender": "female",\n')
        else:
            f.write(f'\t   "gender": "male",\n')
        if prob["hat_avg"]>THRESH_HAT:
            f.write(f'\t   "hat": true,\n')
        else:
            f.write(f'\t   "hat": false,\n')
        if prob["bag_avg"]>THRESH_BAG:
            f.write(f'\t   "bag": true,\n')
        else:
            f.write(f'\t   "bag": false,\n')
        f.write(f'\t   "trajectory": {[int(x) for x in person_lines_crossed.get(id, [])]}\n')
        f.write("\t},\n")
    f.write(']}')
cap.release()
cv2.destroyAllWindows()
