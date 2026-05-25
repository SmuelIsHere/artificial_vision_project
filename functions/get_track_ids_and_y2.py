#ritorna l'id della persona e la coordinata x ed y della parte inferiore del box
def get_track_ids_and_y2(detections):
    track_info = []
    for result in detections:
        for box in result.boxes:
            track_id = box.id if hasattr(box, 'id') else None
            if track_id is not None:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_info.append((track_id, y2, x2))
    return track_info
