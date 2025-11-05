import h5py
import numpy as np


def init_h5_file(path, input_shape, dtype="float16"):
    h5_file = h5py.File(path, "w")
    h5_file.create_dataset("inputs", shape=(0, *input_shape), maxshape=(None, *input_shape),
                           dtype=dtype, chunks=True)
    h5_file.create_dataset("labels", shape=(0,), maxshape=(None,), dtype="int")
    return h5_file

def write_batch_to_h5(h5_file, inputs_list, labels_list):
    """
    Write a batch of inputs and labels into the h5_file at once. 
    Each element in inputs_list should have the same shape, and labels_list should be one-dimensional.
    """
    if not inputs_list:
        return  # empty list, skip
    
    inputs_arr = np.stack(inputs_list, axis=0)  # shape: (batch_size, *input_shape)
    labels_arr = np.array(labels_list, dtype=np.int32)  # shape: (batch_size,)
    
    dset_inputs = h5_file["inputs"]
    dset_labels = h5_file["labels"]
    
    old_len = dset_inputs.shape[0]
    new_len = old_len + inputs_arr.shape[0]
    
    dset_inputs.resize((new_len, *dset_inputs.shape[1:]))
    dset_labels.resize((new_len,))
    
    dset_inputs[old_len:new_len] = inputs_arr
    dset_labels[old_len:new_len] = labels_arr
    
    print(f"  --> Wrote {inputs_arr.shape[0]} samples, total {new_len} so far in {h5_file.filename}")