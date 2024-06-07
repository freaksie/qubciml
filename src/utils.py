import numpy as np

def convert_27bit(x):
    integer,fraction=(divmod(x, 1))
    acc_frac=int(fraction*(2**17))
    fraction_binary=str(bin(acc_frac & 0xFFFFF)[2:].zfill(17))
    if (int(integer)>=0):
        integer_binary=str(bin(int(integer) & 0xFFF)[2:].zfill(10))
    else:
        integer_binary=str(bin(int(integer) & 0b1111111111)[2:].zfill(10))
    binary_value = integer_binary+fraction_binary 
    integer_value = binary_to_integer(binary_value)
    return integer_value
    
def binary_to_integer(binary_string):
    decimal_value = int(binary_string, 2)
    return decimal_value

def convert_18bit(x):
    integer,fraction=(divmod(x, 1))
    acc_frac=int(fraction*(2**12))
    fraction_binary=str(bin(acc_frac & 0xFFFF)[2:].zfill(12))
    if (int(integer)>=0):
        integer_binary=str(bin(int(integer) & 0xFF)[2:].zfill(6))
    else:
        integer_binary=str(bin(int(integer) & 0b111111)[2:].zfill(6))
    binary_value = integer_binary+fraction_binary 
    integer_value = binary_to_integer(binary_value)
    return integer_value