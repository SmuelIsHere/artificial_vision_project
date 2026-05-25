def extract_annotation(label_path):
  annotations_list = []

  with open(label_path, 'r') as file:
    lines = file.readlines()
    for line in lines:

        line = line.strip()
        elements = line.split(',')

        upper = int(elements[1]) if elements[1] != '-1' else -1
        lower = int(elements[2]) if elements[2] != '-1' else -1
        gender = int(elements[3]) if elements[3] != '-1' else -1
        bag = int(elements[4]) if elements[4] != '-1' else -1
        hat = int(elements[5]) if elements[5] != '-1' else -1

        annotation_tuple = (
            elements[0], 
            upper,  
            lower, 
            gender,  
            bag,
            hat
         )

        annotations_list.append(annotation_tuple)

  annotations_list.sort(key=lambda x: x[0])
  return annotations_list
