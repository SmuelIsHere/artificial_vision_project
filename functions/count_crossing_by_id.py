def count_crossing_by_id(track_info1, track_info2, line):
    count = 0
    crossing_ids = []  # Lista per raccogliere gli ID delle persone che hanno attraversato la linea
    
    if line["y1_start"] == line["y1_finish"]:
        for track_id, y1_prev, x1_prev in track_info1:
            for track_id_2, y2_curr, x2_curr in track_info2:
                if track_id == track_id_2:
                    if line["upwards"] == False and (y1_prev < line["y1_start"] and x1_prev > line["x1_start"] and x1_prev < line["x1_finish"]) and (y2_curr > line["y1_start"] and x2_curr > line["x1_start"] and x2_curr < line["x1_finish"]):
                        count += 1
                        crossing_ids.append(track_id)  # Aggiungi l'ID della persona
                    elif line["upwards"] == True and (y1_prev > line["y1_start"] and x1_prev > line["x1_start"] and x1_prev < line["x1_finish"]) and (y2_curr < line["y1_start"] and x2_curr > line["x1_start"] and x2_curr < line["x1_finish"]):
                        count += 1
                        crossing_ids.append(track_id)  # Aggiungi l'ID della persona
    elif line["x1_start"] == line["x1_finish"]:
        for track_id, y1_prev, x1_prev in track_info1:
            for track_id_2, y2_curr, x2_curr in track_info2:
                if track_id == track_id_2:
                    if line["upwards"] == False and (x1_prev < line["x1_start"] and y1_prev > line["y1_start"] and y1_prev < line["y1_finish"]) and (x2_curr > line["x1_start"] and y2_curr > line["y1_start"] and y2_curr < line["y1_finish"]):
                        count += 1
                        crossing_ids.append(track_id)  # Aggiungi l'ID della persona
                    elif line["upwards"] == True and (x1_prev > line["x1_start"] and y1_prev > line["y1_start"] and y1_prev < line["y1_finish"]) and (x2_curr < line["x1_start"] and y2_curr > line["y1_start"] and y2_curr < line["y1_finish"]):
                        count += 1
                        crossing_ids.append(track_id)  # Aggiungi l'ID della persona
    else:
        m = (line["y1_finish"] - line["y1_start"]) / (line["x1_finish"] - line["x1_start"])
        for track_id, y1_prev, x1_prev in track_info1:
            for track_id_2, y2_curr, x2_curr in track_info2:
                if track_id == track_id_2:
                    y_line_prev = line["y1_start"] + m * (x1_prev - line["x1_start"])
                    y_line_curr = line["y1_start"] + m * (x2_curr - line["x1_start"])
                    if line["x1_finish"] > line["x1_start"]:
                        l1 = line["x1_start"]
                        l2 = line["x1_finish"]
                    else:
                        l1 = line["x1_finish"]
                        l2 = line["x1_start"]
                    if  x1_prev > l1 and x1_prev < l2 and x2_curr > l1 and x2_curr < l2:
                        
                        if line["upwards"] == False:
                            if y1_prev < y_line_prev and y2_curr > y_line_curr:
                                count += 1
                                crossing_ids.append(track_id)  # Aggiungi l'ID della persona
                        elif line["upwards"] == True:
                            if y1_prev > y_line_prev and y2_curr < y_line_curr:
                                count += 1
                                crossing_ids.append(track_id)  # Aggiungi l'ID della persona
    
    return count, crossing_ids  # Restituisci sia il conteggio che gli ID
