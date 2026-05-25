person_lines_crossed = {}
#crea la lista delle linee attraversate da ogni persona
def update_lines_crossed(track_id, line_id):
    track_id = int(track_id.item())
    if track_id not in person_lines_crossed:
        person_lines_crossed[track_id] = []
    if line_id not in person_lines_crossed[track_id]:
        person_lines_crossed[track_id].append(line_id)
     