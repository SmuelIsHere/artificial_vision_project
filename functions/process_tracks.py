from functions.draw_line_and_arrow import draw_line_and_arrow
from functions.update_lines_crossed import update_lines_crossed
from functions.count_crossing_by_id import count_crossing_by_id
from functions.update_lines_crossed import person_lines_crossed

import cv2
from torchvision import transforms
from classes.AttributeRecognitionModel import AttributeRecognitionModel
import torch
from PIL import Image

THRESH_HAT = 0.8
THRESH_BAG = 0.4
THRESH_GEN = 0.55

FONT_SCALE_SMALL = 0.3
FONT_THICKNESS_SMALL = 1
FONT_SCALE_ID = 0.4
FONT_THICKNESS_ID = 1
BOX_PADDING = 6
FRAME_WAIT_BEFORE_PAR = 10
ATTRIBUTE_MODEL_PATH = 'par_models/unfr_best_loss.pth'

attribute_model = AttributeRecognitionModel(num_attributes=3)
attribute_model.load_state_dict(torch.load(ATTRIBUTE_MODEL_PATH, map_location=torch.device('cpu')))
attribute_model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

id_in_scene_counter = {} # conta quante volte ogni persona è nel frame

def process_tracks(results, track_info1, track_info2, resized_frame, resized_frame_no_boxes,
                    total_people_set, line_specs, line_counts, ids_classification_story):
    
    for _,line in enumerate(line_specs):
        idx = line["id"]
        resized_frame = draw_line_and_arrow(resized_frame, line, line_color=(255, 0, 0))
    
        if track_info1 is not None:
            people_crossed, crossing_ids = count_crossing_by_id(track_info1, track_info2, line)
            line_counts[idx-1] += people_crossed
            for track_id in crossing_ids:
                update_lines_crossed(track_id, idx)
                
    for track in results[0].boxes:
            box, track_id = track.xyxy[0], track.id if hasattr(track, 'id') else None

            if track_id is not None:

                # Conto per quanti frame una persona compare in scena
                if track_id.item() not in id_in_scene_counter:
                    id_in_scene_counter.update({track_id.item() : 0})
                elif id_in_scene_counter[track_id.item()] < FRAME_WAIT_BEFORE_PAR:
                    id_in_scene_counter[track_id.item()] += 1
                
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                total_people_set.add(int(track_id.item()))
                label_size, _ = cv2.getTextSize(str(int(track_id.item())),
                                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_ID, FONT_THICKNESS_ID)
                label_w, label_h = label_size

                top_left = (x1, y1)
                bottom_right = (x1 + label_w + 4, y1 + label_h + 4)
                cv2.rectangle(resized_frame, top_left, bottom_right, (255, 255, 255), -1)

                label = f"{int(track_id.item())}"

                gender_label = ""
                bag_label = ""
                hat_label = ""

                #verifico se ho già fatto il par più di un certon numero di volte
                if ids_classification_story.get_probabilities_count(track_id.item()) < FRAME_WAIT_BEFORE_PAR:
                    # PAR
                    roi = resized_frame_no_boxes[y1:y2, x1:x2]
                    roi_pil = Image.fromarray(roi)
                    input_tensor = transform(roi_pil).unsqueeze(0)
                    predictions = attribute_model(input_tensor)

                    ids_classification_story.add_probabilities(track_id.item(), predictions[0], predictions[2], predictions[1])
                    
                else:
                    avg_prob_dict = ids_classification_story.get_average_probabilities(track_id.item())
                    avg_prob_list = []
                    avg_prob_list.append(avg_prob_dict['gender_avg'])
                    avg_prob_list.append(avg_prob_dict['bag_avg'])
                    avg_prob_list.append(avg_prob_dict['hat_avg'])

                    for i, pred in enumerate(avg_prob_list):

                        if(i == 0):
                            if(pred.item() > THRESH_GEN):
                                gender_label = "Gender: F"
                            else:
                                gender_label ="Gender: M"
                            continue
                        
                        if(i == 1):
                            if(pred.item() > THRESH_BAG):
                                bag_label = "Bag "
                            else:
                                bag_label = "No Bag "
                            continue

                        if(i == 2):
                            if(pred.item() > THRESH_HAT):
                                hat_label = "Hat "
                            else:
                                hat_label = "No Hat "
                            continue
                    
                lines_crossed_text = f"[{', '.join(map(str, person_lines_crossed.get(track_id.item(), [])))}]"

                cv2.putText(resized_frame, label, (x1 + 2, y1 + label_h + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_ID, (0, 0, 255), FONT_THICKNESS_ID)
                
                info_lines = [gender_label, (bag_label+hat_label), f"{lines_crossed_text}"]
                text_sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                               FONT_SCALE_SMALL, FONT_THICKNESS_SMALL)[0] for line in info_lines]
                
                box_width = max([size[0] for size in text_sizes]) + 6  
                box_height = sum([size[1] for size in text_sizes]) + len(text_sizes) * 4  
                info_box_top_left = (x1, y2 + 2)
                info_box_bottom_right = (x1 + box_width, y2 + 2 + box_height)
                cv2.rectangle(resized_frame, info_box_top_left, info_box_bottom_right, (255, 255, 255), -1)
                
                y_offset = y2 + 10 
                for line in info_lines:
                    cv2.putText(resized_frame, line, (x1 + 2, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, (0, 0, 0), FONT_THICKNESS_SMALL)
                    y_offset += text_sizes[0][1] + 4  
    return resized_frame, ids_classification_story
