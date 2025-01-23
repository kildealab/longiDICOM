import json

def save_to_json(imaging_data,location):
    with open(location, "w") as write_file:
        json.dump(imaging_data, write_file, indent=4)