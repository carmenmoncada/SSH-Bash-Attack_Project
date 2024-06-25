import numpy as np
import json
from pathlib import Path
def linear_to_db(x):
    return 10*np.log10(x)

def db_to_linear(x):
    return 10**(x/10)

# Function used in the elements.py to compute the number of channel that will be use in the code
# It read the json file and according to the switching matrix defined in it
# The number of channel will be computed, if there is no 'swithcing_matrix' key, it assigns 10 channels
def compute_number_of_channel(filename):
    # Convert filename to Path object if it's not already
    if not isinstance(filename, Path):
        filename = Path(filename)

    # Check if the file exists
    if not filename.exists():
        raise FileNotFoundError(f"File '{filename}' not found.")

    # Read the content of the file
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file '{filename}': {e}")
        return None  # Handle the error appropriately
    # Access the first key in the main dictionary
    first_key = next(iter(data))
    # Access the dictionary corresponding to the first key
    first_element = data[first_key]

    if 'switching_matrix' in first_element:
        # Accessing the 'switching_matrix' dictionary
        switching_matrix_info = data[first_key]['switching_matrix']
        # Retrieving the first key in the 'switching_matrix' dictionary
        first_key_switching_matrix = next(iter(switching_matrix_info))
        # Access the list associated with the first key of the first element
        first_element_list = switching_matrix_info[first_key_switching_matrix][first_key_switching_matrix]
        # Get the length of the list
        number_of_channels = len(first_element_list)
        return number_of_channels
    else:
        number_of_channels = 10
        return number_of_channels
