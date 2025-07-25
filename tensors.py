import numpy as np
import torch

def main():
    #normal python lists
    my_list = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]

    #numpy array
    np1 = np.random.rand(3, 4)
    print(np1)
    print(np1.dtype)

    #tensors
    tensor_2d = torch.randn(3, 4)
    print(tensor_2d)

    tensor_3d = torch.zeros(2, 3, 4)
    print(tensor_3d)

    #create tensor out of numpy array
    my_tensor = torch.tensor(np1)
    print(my_tensor) #default type is float64 now cuz numpy array is float64

    # tensor ops
    # torch.arange(start, end, default spacing)
    # returns a list of numbers from start to end, step is default spacing or user-defined
    my_torch = torch.arange(10)
    print(my_torch)

    my_torch = my_torch.reshape(2, 5) #creates 2d tensor to fit
    print(my_torch)

    #reshape if don't know num of items using -1
    my_torch2 = torch.arange(10)
    my_torch2 = my_torch2.reshape(2, -1)

    my_torch3 = torch.arange(10)
    my_torch4 = my_torch3.view(2, 5) #view and reshape very difference
    print(my_torch4)

    #reshape and view update - memory aliasing
    my_torch5 = torch.arange(10)
    my_torch6 = my_torch5.reshape(2, 5)
    print(my_torch6)
    my_torch5[1] = 4141 #note, if you do this on a reshape, my_torch5 will point to the second 2d array (bc ptr arith),
    # will view differently and thus will replace the entire second 2d array with 4141. stupid python
    print(my_torch5)
    print(my_torch6)

    #tensor slices
    my_torch7 = torch.arange(10)
    print(my_torch7[7])
    my_torch8 = my_torch7.reshape(5, 2)
    print(my_torch8)
    print(my_torch8[:,1]) #get stuff up until 2nd column

    #return column
    print(my_torch8[:,1:]) #this keeps structure



if __name__ == "__main__":
        main()