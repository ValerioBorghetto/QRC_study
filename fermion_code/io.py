import json


def write_json(file, data):
    """
    Writes a dictionary into a JSON file.
    
    Arguments:
        file: file path (string)
        data: dictionary to write
    """
    with open(file, 'w') as f:
        json.dump(data, f, indent=2)


def read_json(file):
    """
    Reads a dictionary from a JSON file.
    
    Arguments:
        file: file path (string)
    
    Returns:
        Dictionary loaded from JSON
    """
    with open(file, 'r') as f:
        return json.load(f)
