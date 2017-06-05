import numpy as np
from notebook_helpers import make_grid_svg, draw_strokes, encode

def dim_sort(dim, zs):
    ixs = zs[:,dim].argsort()
    return ixs
    
def batch_zs(sketches, model, sess):
    res = np.zeros( (len(sketches), 128) )
    for i, sk in enumerate(sketches):
        res[i] = encode(sk, model, sess)
    return res
    
def sort_and_draw(dim, sketches, zs):
    ixs = dim_sort(dim, zs)
    cols = 8
    grid = []
    for rank, i in enumerate(ixs):
        grid.append([
            sketches[i],
            [rank/cols, rank%cols]
        ])
    strokes = make_grid_svg(grid, grid_space=16.0, grid_space_x=24.0)
    draw_strokes(strokes)
    
