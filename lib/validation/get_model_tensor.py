import sys
import os.path as osp

import tensorflow as tf
file_name = osp.join(osp.dirname(__file__), '../..', 'out', 'sub_sample', 'model.ckpt-160001') #file_name: Name of the checkpoint file.
assert osp.exists(file_name)

def print_tensors_in_checkpoint_file(tensor_name):
    """Prints tensors in a checkpoint file.
    If no `tensor_name` is provided, prints the tensor names and shapes
    in the checkpoint file.
    If `tensor_name` is provided, prints the content of the tensor.
    Args:

    tensor_name: Name of the tensor in the checkpoint file to print.
    """
    try:
        reader = tf.train.NewCheckpointReader(file_name)
        if not tensor_name:
            print(reader.debug_string().decode("utf-8"))
        else:
            print("tensor_name: ", tensor_name)
            return reader.get_tensor(tensor_name)
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                "with SNAPPY.")
if __name__=="__main__":
    print_tensors_in_checkpoint_file("embedding:0")