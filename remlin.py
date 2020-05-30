#!/usr/bin/python3
import sys
import numpy as np
from PIL import Image

# arr: greyscale 2d np arrays from Image.convert("L")
# returns: arr but blurred. Each pixel is taken max brightness with 4 random pixels on same column.
def VerticalBlur(arr):
  magic_shifts = [1071, 2847, 3251, 4933]
  new_arr = arr
  for shift in magic_shifts:
    new_arr = np.maximum(new_arr, np.roll(arr, shift, axis=0))
  return new_arr

# arr: greyscale 2d np arrays from Image.convert("L"), blurred or not
# returns: consecutive range of ints (width in total), indicating column numbers of the found vert line
def FindVertLine(arr, width):
  _, ncols = arr.shape
  min_start = -1
  min_sum = sys.maxsize
  accu = np.sum(arr, axis=0)
  for start in range(0, ncols-width+1):
    the_sum = np.sum(accu[start:start+width])
    if the_sum < min_sum:
      min_sum = the_sum
      min_start = start
  return range(min_start, min_start+width)

def TagVertLine(arr, col_indices, rgb):
  arr[:, np.array(col_indices)] = np.array(rgb)
