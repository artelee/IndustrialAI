import os, glob
from collections import defaultdict

GALLERY_DIR = '/home/ubuntu/datasets/market1501/Market-1501-v15.09.15/bounding_box_test'
QUERY_DIR = '/home/ubuntu/datasets/market1501/Market-1501-v15.09.15/query'

def parse(f):
    p = os.path.basename(f).split('_')
    return p[0], p[1][:2]

gallery_by_id = defaultdict(set)
for f in sorted(glob.glob(f'{GALLERY_DIR}/*.jpg')):
    pid, cam = parse(f)
    if pid in ('-1','0000'): continue
    gallery_by_id[pid].add(cam)

query_by_id = defaultdict(set)
for f in sorted(glob.glob(f'{QUERY_DIR}/*.jpg')):
    pid, cam = parse(f)
    query_by_id[pid].add(cam)

# c1 있는 ID
c1_ids = [pid for pid, cams in gallery_by_id.items() if 'c1' in cams]
print(f'c1에 갤러리 있는 ID: {len(c1_ids)}명')

# c1 + c6 query 있는 ID
c1_c6q = [pid for pid in c1_ids if 'c6' in query_by_id.get(pid, set())]
print(f'c1 갤러리 + c6 query 있는 ID: {len(c1_c6q)}명')

# c1 + 다른 카메라 query 있는 ID
for cam in ['c2','c3','c4','c5','c6']:
    n = sum(1 for pid in c1_ids if cam in query_by_id.get(pid, set()))
    print(f'c1 갤러리 + {cam} query 있는 ID: {n}명')