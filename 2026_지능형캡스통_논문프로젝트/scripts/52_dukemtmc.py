import os, glob
from collections import defaultdict

DUKE = '/home/ubuntu/datasets/dukemtmc-reid/DukeMTMC-reID'
GALLERY = f'{DUKE}/bounding_box_test'
QUERY = f'{DUKE}/query'

def parse(f):
    p = os.path.basename(f).split('_')
    return p[0], p[1][:2]

gallery_by_id = defaultdict(set)
for f in sorted(glob.glob(f'{GALLERY}/*.jpg')):
    pid, cam = parse(f)
    if pid in ('-1','0000'): continue
    gallery_by_id[pid].add(cam)

query_by_id = defaultdict(set)
for f in sorted(glob.glob(f'{QUERY}/*.jpg')):
    pid, cam = parse(f)
    query_by_id[pid].add(cam)

print(f'총 갤러리 ID: {len(gallery_by_id)}')
print(f'총 query ID: {len(query_by_id)}')

# c1 있는 ID
c1_ids = [pid for pid, cams in gallery_by_id.items() if 'c1' in cams]
print(f'c1 갤러리 있는 ID: {len(c1_ids)}')

# 각 카메라별 query
all_cams = set()
for cams in gallery_by_id.values():
    all_cams.update(cams)
print(f'카메라: {sorted(all_cams)}')

for cam in sorted(all_cams):
    if cam == 'c1': continue
    n = sum(1 for pid in c1_ids if cam in query_by_id.get(pid, set()))
    print(f'c1 갤러리 + {cam} query: {n}명')
