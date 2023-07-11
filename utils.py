from scipy import ndimage
import numpy as np

def get_largest_element(comp, thr=0.1, minsize=None, outlabels=False):
    tot = np.sum(comp > 0)
    labels,num = ndimage.label(comp, structure=ndimage.generate_binary_structure(comp.ndim, 1))
    hist,bins = np.histogram(labels, bins=num, range=(1,num+1))
    argsort_hist = np.argsort(hist)[::-1]

    if minsize is None:
        minsize = np.max(hist) + 1

    where = np.where((hist/tot > thr) | (hist > minsize))[0] + 1
    print(num,'components\t',len(where),'preserved')
    print(np.sort(hist)[::-1][:20])

    mask = labels == where[0]
    for w in where[1:]:
        mask = mask | (labels == w)
    box0 = comp.copy()
    box0[~mask] = 0

    if outlabels:
        return box0, labels, where

    return box0