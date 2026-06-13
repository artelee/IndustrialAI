import os, glob
from collections import defaultdict

OCC = '/home/ubuntu/datasets/occluded_duke'
def parse(f):
    p = os.path.basename(f).split('_')
    return p[0], p[1]

g = defaultdict(set)
for f in glob.glob(f'{OCC}/bounding_box_test/*.jpg'):
    pid, cam = parse(f)
    if pid not in ('-1','0000'):
        g[pid].add(cam)

q = defaultdict(set)
for f in glob.glob(f'{OCC}/query/*.jpg'):
    pid, cam = parse(f)
    q[pid].add(cam)

c1_ids = [pid for pid in g if 'c1' in g[pid]]
print(f'c1 갤러리 ID: {len(c1_ids)}')
for cam in sorted(set(c for cams in g.values() for c in cams)):
    if cam == 'c1': continue
    n = sum(1 for pid in c1_ids if cam in q.get(pid, set()))
    print(f'c1 + {cam} query: {n}명')
