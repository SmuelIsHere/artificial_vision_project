import os
def load_line_specs():
    txt_file = f"line_specs.txt"
    if not os.path.exists(txt_file):
        print(f"Error: {txt_file} not found!")
        return []
    line_specs = []
    with open(txt_file, "r") as file:
        for line in file:
            try:
                line_data = eval(line.strip())
                if isinstance(line_data, dict):
                    line_specs.append(line_data)
            except Exception as e:
                print(f"Error parsing line in {txt_file}: {e}")
    return line_specs
