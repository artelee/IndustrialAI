import torch
w = torch.load('checkpoints/clipreid_duke_nosie.pth', map_location='cpu')
print('타입:', type(w))
if isinstance(w, dict):
    keys = list(w.keys())
    print('총 키 수:', len(keys))
    print()
    # position embedding 관련
    for k in keys:
        if 'pos_embed' in k.lower() or 'positional' in k.lower():
            print('POS:', k, '->', tuple(w[k].shape))
    print()
    # SIE / cv_embed 관련 (카메라 임베딩)
    for k in keys:
        if 'sie' in k.lower() or 'cv_embed' in k.lower() or 'cam' in k.lower():
            print('SIE?:', k, '->', tuple(w[k].shape))
    print()
    # 앞 10개 키 구조 파악
    print('--- 처음 15개 키 ---')
    for k in keys[:15]:
        print(k, tuple(w[k].shape) if hasattr(w[k],'shape') else type(w[k]))
