def numpy_array_to_list_of_arrays(array):
    '''Used to return a numpy array a as a list in order to put it into the DataFrame. 
    Instead of calling .tolist() which creates a nested list for the entire numpy array, 
    this procedure just creates a list for the outermost dimension, i.e., the result of 
    this function is to return a list of numpy arrays.'''
    return [item for item in array]
