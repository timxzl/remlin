#!/usr/bin/python3
import os
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

def Pad(arr, row_pad, col_pad):
  avg = np.mean(arr, axis=(0,1)).astype(int)
  vert_shape = list(arr.shape)
  vert_shape[0] = row_pad
  vert_pad = np.empty_like(arr, shape=vert_shape)
  vert_pad[:,:] = avg
  arr = np.concatenate((vert_pad, arr, vert_pad), axis=0)

  hor_shape = list(arr.shape)
  hor_shape[1] = col_pad
  hor_pad = np.empty_like(arr, shape=hor_shape)
  hor_pad[:,:] = avg
  return np.concatenate((hor_pad, arr, hor_pad), axis=1)

def PatchDist(a, b, masked_cols = None):
  diff = a-b
  if masked_cols is not None:
    diff[:,masked_cols] = 0
  return np.sum(np.square(diff))

def MatchPatch(patch, arr, row0, col0, row_range, col_range, masked_cols = None):
  shift_penalty = 2000
  mindist = sys.maxsize
  min_row, min_col = row0, col0
  min_shift = 0
  nrows = patch.shape[0]
  ncols = patch.shape[1]
  for drow in row_range:
    for dcol in col_range:
      row = row0 + drow
      col = col0 + dcol
      shift = drow*drow+dcol*dcol
      dist = PatchDist(patch, arr[row : row+nrows, col : col+ncols], masked_cols) + shift*shift_penalty
      if dist < mindist or (dist == mindist and shift < min_shift):
        mindist = dist
        min_shift = shift
        min_row, min_col = row, col
  return (min_row, min_col, mindist)

# stripe: a vertical subrange of graph
# arr must already be patched top and bottom, at least cover row_range
def MatchStripe(arr, patch_cols, step, recover_src_arr, row0, col0, row_range, col_range,
    masked_cols = None,
    recover = False,
    colors = [(1, 200, 250), (250, 1, 200), (200, 250, 1)]):
  tagged_arr = None
  tagged_recover_src_arr = None
  if colors is not None:
    tagged_arr = np.copy(arr)
    tagged_recover_src_arr = np.copy(recover_src_arr)

  color_idx = 0
  col_start = min(patch_cols)
  col_end = max(patch_cols)+1
  nrows = arr.shape[0]
  ncols = len(patch_cols)
  result = []
  top = 0
  while top < nrows:
    bottom = top + step
    if bottom > nrows:
      bottom = nrows
    patch = arr[top:bottom, patch_cols]
    min_row, min_col, mindist = MatchPatch(patch, recover_src_arr, row0, col0, row_range, col_range, masked_cols)

    result.append((min_row, min_col, mindist))
    print("Match", top, ":", bottom, "is", min_row, min_col, mindist, "vs row0,col0:", row0, col0)

    if masked_cols is not None and recover:
      arr[top : bottom, masked_cols+col_start] = recover_src_arr[min_row : min_row+(bottom-top), masked_cols+min_col]

    if colors is not None:
      TagRect(tagged_arr, top, bottom, col_start, col_end, colors[color_idx])
      TagRect(tagged_recover_src_arr, min_row, min_row+step, min_col, min_col+(col_end-col_start+1), colors[color_idx])
      color_idx = (color_idx+1)%len(colors)

    # Use this patch's position to calculate next patch's initial estimate position
    row0 = min_row+step
    col0 = min_col
    top = bottom

  return (result, tagged_arr, tagged_recover_src_arr)

def RemoveVertLine(arr, recover_src_arr, line_width, step, patch_extend, row_range, col_range):
  masked_cols = FindVertLine(VerticalBlur(arr), line_width)
  print("Columns masked by a vertical line:", masked_cols)

  row_pad = 100+max([abs(x) for x in row_range])*2
  col_pad = 100+max([abs(x) for x in col_range])*2
  padded_src = Pad(recover_src_arr, row_pad, col_pad)
  patch_cols = range(min(masked_cols)-patch_extend, max(masked_cols)+patch_extend)
  row0 = row_pad
  col0 = col_pad + min(patch_cols)
  result = MatchStripe(arr, patch_cols, step, padded_src, row0, col0, row_range, col_range, masked_cols = np.array(range(patch_extend, patch_extend+line_width)), recover=True)
  return (result[1], result[2])

def RemLin(img, recover_src_img, step, patch_extend):
  line_width = 4
  row_range = range(-50, 50)
  col_range = range(-50, 50)

  arr = np.array(img)
  recover_src_arr = np.array(recover_src_img)
  tagged_arr, tagged_recover_src = RemoveVertLine(arr, recover_src_arr, line_width, step, patch_extend, row_range, col_range)
  return (Image.fromarray(arr), Image.fromarray(tagged_arr), Image.fromarray(tagged_recover_src))

def WriteImg(img, path):
  with open(path, "wb") as f:
    img.save(f)

def MainProcess(path_img, path_recover_src):
  out_path, _ = os.path.splitext(path_img)
  img = Image.open(path_img)
  recover_src_img = Image.open(path_recover_src)
  recovered_img, tagged_img, tagged_recover_src = RemLin(img, recover_src_img, 220, 150)
  WriteImg(recovered_img, out_path + "_recovered_img_.png")
  WriteImg(tagged_img, out_path + "_tagged_original_img_.png")
  WriteImg(tagged_recover_src, out_path + "_tagged_recover_src_img_.png")

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage:", sys.argv[0], "image_to_remove_vert_line another_image_without_vert_line")
  else:
    MainProcess(sys.argv[1], sys.argv[2])
