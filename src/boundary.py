from typing import Iterable
import numpy as np
from typing import Iterable, Tuple

Chain = Iterable[int]

def build_chains(
    segs: Iterable[Chain],
    finished: Iterable[Chain]
)-> Tuple[Iterable[Chain], Iterable[Chain]]:
    if len(segs) == 0: 
        return segs, finished 
    if len(segs) == 1: 
        finished.extend(segs) 
        return [], finished
    left_segs = sorted(segs, key=lambda x: x[-1])
    right_segs = sorted(segs, key=lambda x: x[0])
    idx_rl = np.argsort(np.array([x[0] for x in left_segs]))
    ret = []
    added = np.zeros(len(left_segs), dtype=np.bool8)
    for i, (l_seg, r_seg) in enumerate(zip(left_segs, right_segs)):
        if (not added[i]) and l_seg[0] == l_seg[-1]:
            added[i] = True
            finished.append(l_seg)
        elif (not added[i]) and (not added[idx_rl[i]]):
            added[i] = True
            added[idx_rl[i]] = True
            ret.append((*l_seg, *r_seg))
    for seg, was_added in zip(left_segs, added):
        if not was_added:
            ret.append(seg)
    return build_chains(ret, finished)


xs = [(4, 0),
    (0, 1),
    (1, 2),
    (2, 3),
    (9, 4),
    (6, 5),
    (7, 6),
    (11, 7),
    (3, 8),
    (13, 9),
    (5, 10),
    (10, 11),
    (8, 12),
    (14, 13),
    (15, 14),
    (12, 15)]

print(build_chains(xs, []))