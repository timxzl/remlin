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
  ncols = arr.shape[1]
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

def TagRect(arr, top, bottom, left, right, rgb):
  arr[top:top+3, left : right] = rgb
  arr[bottom-3:bottom, left : right] = rgb
  arr[top : bottom, left:left+3] = rgb
  arr[top : bottom, right-3:right] = rgb

def VertPad(arr, top_rows, bottom_rows):
  avg = np.mean(arr, axis=(0,1)).astype(int)
  top_shape = list(arr.shape)
  top_shape[0] = top_rows
  top = np.empty_like(arr, shape=top_shape)
  top[:,:] = avg
  bottom_shape = list(arr.shape)
  bottom_shape[0] = bottom_rows
  bottom = np.empty_like(arr, shape=bottom_shape)
  bottom[:,:] = avg
  return np.concatenate((top, arr, bottom), axis=0)

def PatchDist(a, b, masked_cols = None):
  diff = a-b
  if masked_cols is not None:
    diff[:,masked_cols] = 0
  return np.sum(np.square(diff))

def MatchPatch(patch, arr, row0, col0, row_range, col_range, masked_cols = None):
  mindist = sys.maxsize
  min_row, min_col = row0, col0
  min_shift = 0
  nrows = patch.shape[0]
  ncols = patch.shape[1]
  for drow in row_range:
    for dcol in col_range:
      row = row0 + drow
      col = col0 + dcol
      dist = PatchDist(patch, arr[row : row+nrows, col : col+ncols], masked_cols)
      shift = drow*drow+dcol*dcol
      if dist < mindist or (dist == mindist and shift < min_shift):
        mindist = dist
        min_shift = shift
        min_row, min_col = row, col
  return (min_row, min_col, mindist)

# stripe: a vertical subrange of graph
# arr must already be patched top and bottom, at least cover row_range
def MatchStripe(stripe, step, arr, row0, col0, row_range, col_range, masked_cols = None,
    colors = [(1, 200, 250), (250, 1, 200), (200, 250, 1)]):
  tagged_arr = None
  if colors is not None:
    tagged_arr = np.copy(arr)

  color_idx = 0
  nrows = stripe.shape[0]
  ncols = stripe.shape[1]
  result = []
  top = 0
  while top < nrows:
    bottom = top + step
    if bottom > nrows:
      bottom = nrows
    patch = stripe[top:bottom]
    min_row, min_col, mindist = MatchPatch(patch, arr, row0, col0, row_range, col_range, masked_cols)
    result.append((min_row, min_col, mindist))
    print("Match", top, ":", bottom, "is", min_row, min_col, mindist)
    if colors is not None:
      TagRect(tagged_arr, min_row, min_row+step, min_col, min_col+ncols, colors[color_idx])
      TagRect(stripe, top, bottom, 0, ncols, colors[color_idx])
      color_idx = (color_idx+1)%len(colors)
    # Use this patch's position to calculate next patch's initial estimate position
    row0 = min_row+step
    col0 = min_col
    top = bottom

  return (result, tagged_arr)
