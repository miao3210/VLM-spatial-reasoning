import pickle
import json
import scipy.io as sio

def save_to_pickle(data, filename):
    """
    Save data to a pickle file.
    
    Args:
        data: The data to save.
        filename: The name of the file to save the data to.
    """
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_from_pickle(filename):
    """
    Load data from a pickle file.
    
    Args:
        filename: The name of the file to load the data from.
    
    Returns:
        The loaded data.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)
    
def save_to_json(data, filename):
    """
    Save data to a JSON file.
    
    Args:
        data: The data to save.
        filename: The name of the file to save the data to.
    """
    with open(filename, "w") as f:
        json.dump(data, f)

def load_from_json(filename):
    """
    Load data from a JSON file.
    
    Args:
        filename: The name of the file to load the data from.
    
    Returns:
        The loaded data.
    """
    with open(filename, "r") as f:
        return json.load(f)
    
def save_to_mat(data, filename):
    """
    Save data to a MATLAB .mat file.
    
    Args:
        data: The data to save.
        filename: The name of the file to save the data to.
    """
    sio.savemat(filename, data)

def load_from_mat(filename):
    """
    Load data from a MATLAB .mat file.
    
    Args:
        filename: The name of the file to load the data from.
    
    Returns:
        The loaded data.
    """
    return sio.loadmat(filename, squeeze_me=True, struct_as_record=False)