#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
111_query_transform_eval.py ─ Query-side 포즈 변환 ReID 평가 (은혜 원래 설계)

[설계]
  Baseline:  Query = C1 real,                       Gallery = C6 real(전체)
  Proposed:  Query = C1 인물 + C6 medoid 포즈 생성본,  Gallery = C6 real(전체, 동일)

  생성 방향: IPA(외형)=C1 query 이미지, ControlNet(포즈)=C6 medoid skeleton
            → "C6 포즈를 취한 C1 사람" 을 query feature 로 사용
  Gallery 는 양쪽 동일 → 정답 집합 불변 → mAP 자해(110번 -27 함정) 없음

[비교]
                      보정X        보정O(CP)
  Baseline (C1 real)   ──           ──
  Proposed (생성 query) ──           ──
  Proposed+열화         ──           ──
"""

import os, sys, glob, io, gc, torch, numpy as np
from collections import defaultdict
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
import torch.nn as nn
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux import OpenposeDetector

# ===== CONFIG =====
HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
GEN_DIR = f"{PROJECT_DIR}/outputs/c1query_to_c6pose"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"
LOG = f"{PROJECT_DIR}/EXPERIMENT_LOG.md"

device, dtype = "cuda", torch.float16
QUERY_CAM = "c1"          # query 카메라 (변환 대상)
GALLERY_CAM = "c6"        # gallery 카메라 (목표 시점)
POSE_POOL_SIZE = 100
SEED, SIZE = 42, (384, 768)
IPA_SCALE, CN_SCALE = 0.8, 1.0

os.makedirs(f"{GEN_DIR}/raw", exist_ok=True)
os.makedirs(f"{GEN_DIR}/matched", exist_ok=True)

def logline(m):
    with open(LOG, "a") as fp: fp.write(m + "\n")
    print(m)
def parse(f):
    p = os.path.basename(f).split("_"); return p[0], p[1][:2]
def cam_idx(tc): return int(tc[1:])-1


# ===== 1. 데이터 =====
print("데이터 로드...")
gallery_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{GALLERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    if pid in ('-1', '0000'): continue
    gallery_by_id[pid][cam].append(f)
query_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{QUERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    query_by_id[pid][cam].append(f)

# Query 대상: C1 query 있고 + C6 gallery 에 정답 있는 사람 (그래야 평가 가능)
query_ids = [pid for pid in sorted(query_by_id)
             if QUERY_CAM in query_by_id[pid]
             and GALLERY_CAM in gallery_by_id[pid]]
print(f"Query 대상 ID(C1 query & C6 정답 보유): {len(query_ids)}명")

# Gallery: C6 에 등장하는 모든 사람 (distractor 포함, 표준)
gallery_ids = [pid for pid in sorted(gallery_by_id) if GALLERY_CAM in gallery_by_id[pid]]
print(f"Gallery ID(C6 전체, distractor 포함): {len(gallery_ids)}명\n")


# ===== 2. C6 medoid 포즈 =====
print("OpenPose 로드 + C6 medoid 포즈...")
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)
def detect_skel(img):
    try:
        o = openpose(img)
        if isinstance(o, tuple): o = o[0]
        if o is None: return None
        if not isinstance(o, Image.Image): o = Image.fromarray(o)
        return o
    except Exception:
        return None
def select_medoid(skels):
    small = np.stack([np.array(s.resize((48,96))).astype(np.float32).flatten() for s in skels])
    small = small/(np.linalg.norm(small,axis=1,keepdims=True)+1e-8)
    return int((small@small.T).mean(axis=1).argmax())

q_set = set(query_ids)
pool=[]
for pid in gallery_ids:
    if pid in q_set: continue          # query 인물 제외 (자세 누출 방지)
    pool.append(gallery_by_id[pid][GALLERY_CAM][0])
    if len(pool) >= POSE_POOL_SIZE*2: break
skels=[]
for p in pool:
    s=detect_skel(Image.open(p).convert("RGB").resize(SIZE, Image.LANCZOS))
    if s is not None: skels.append(s)
    if len(skels)>=POSE_POOL_SIZE: break
c6_medoid_skel = skels[select_medoid(skels)].resize(SIZE, Image.LANCZOS)
print(f"  C6 medoid pose 확보 ({len(skels)}장 중)\n")


# ===== 3. 열화 함수 =====
def analyze_cctv(ref):
    a=np.asarray(ref.convert("RGB")).astype(np.float32); gray=a.mean(2)
    lap=np.abs(np.gradient(np.gradient(gray,axis=0)[0],axis=0)).var()
    blurred=Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
    noise=float((gray-np.asarray(blurred)).std())
    return dict(sharp=float(lap),noise=noise,bright=float(gray.mean()),
                contrast=float(gray.std()),
                sat=float(np.asarray(ref.convert("HSV")).astype(np.float32)[:,:,1].mean()))
def degrade_matched(img, ref):
    s=analyze_cctv(ref); w,h=img.size
    down=int(np.clip(48+s["sharp"]*0.5,40,96))
    img=img.resize((down,int(down*h/w)),Image.BILINEAR).resize((w,h),Image.BILINEAR)
    img=img.filter(ImageFilter.GaussianBlur(float(np.clip(1.2-s["sharp"]*0.01,0.3,1.2))))
    g=np.asarray(img.convert("RGB")).astype(np.float32).mean(2)
    img=ImageEnhance.Brightness(img).enhance(float(np.clip(s["bright"]/(g.mean()+1e-6),0.7,1.3)))
    g2=np.asarray(img.convert("RGB")).astype(np.float32).mean(2)
    img=ImageEnhance.Contrast(img).enhance(float(np.clip(s["contrast"]/(g2.std()+1e-6),0.7,1.3)))
    cs=np.asarray(img.convert("HSV")).astype(np.float32)[:,:,1].mean()
    img=ImageEnhance.Color(img).enhance(float(np.clip(s["sat"]/(cs+1e-6),0.7,1.3)))
    arr=np.asarray(img.convert("RGB")).astype(np.float32)
    arr+=np.random.RandomState(SEED).normal(0,s["noise"]*0.6,arr.shape)
    buf=io.BytesIO(); Image.fromarray(np.clip(arr,0,255).astype(np.uint8)).save(buf,"JPEG",quality=55)
    buf.seek(0); return Image.open(buf).convert("RGB")


# ===== 4. 생성: C1 query 인물 → C6 포즈 =====
print("생성 파이프라인 로드 (txt2img + IPA)...")
cn=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose",
                                   cache_dir=CACHE_DIR, torch_dtype=dtype)
pipe=StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=cn, cache_dir=CACHE_DIR, torch_dtype=dtype,
    safety_checker=None, requires_safety_checker=False)
pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                     weight_name="ip-adapter-plus_sd15.safetensors")
pipe.set_ip_adapter_scale(IPA_SCALE)
pipe=pipe.to(device)
print("로드 완료. 생성 시작 (C1 인물 → C6 포즈)...")

for i, pid in enumerate(query_ids):
    c1_img=Image.open(sorted(query_by_id[pid][QUERY_CAM])[0]).convert("RGB")  # C1 query 외형
    raw_p=f"{GEN_DIR}/raw/{pid}_gen.png"
    mat_p=f"{GEN_DIR}/matched/{pid}_gen.png"
    if os.path.exists(raw_p) and os.path.exists(mat_p): continue
    g=torch.Generator(device).manual_seed(SEED)
    gen=pipe(prompt="a photo of a person, full body, surveillance",
             negative_prompt="blurry, low quality, deformed, multiple people, extra limbs",
             image=c6_medoid_skel,          # 포즈 = C6 medoid
             ip_adapter_image=c1_img,       # 외형 = C1 query 인물
             controlnet_conditioning_scale=CN_SCALE,
             num_inference_steps=30, guidance_scale=7.5,
             width=SIZE[0], height=SIZE[1], generator=g).images[0]
    gen.save(raw_p)
    degrade_matched(gen, c1_img).save(mat_p)
    if (i+1)%20==0: print(f"  {i+1}/{len(query_ids)}")
del pipe, cn; gc.collect(); torch.cuda.empty_cache()
print("생성 완료\n")


# ===== 5. CLIP-ReID =====
print("CLIP-ReID 로드...")
sys.path.insert(0, "/home/ubuntu/CLIP-ReID")
from config import cfg as rcfg
from model.make_model_clipreid import make_model
W=f"{CACHE_DIR}/clipreid_duke_nosie.pth"
rcfg.MODEL.NAME="ViT-B-16"; rcfg.MODEL.STRIDE_SIZE=[16,16]
rcfg.MODEL.SIE_CAMERA=False; rcfg.MODEL.SIE_COE=0.0; rcfg.MODEL.ID_LOSS_TYPE="softmax"
rcfg.INPUT.SIZE_TRAIN=[256,128]; rcfg.INPUT.SIZE_TEST=[256,128]
rcfg.INPUT.PIXEL_MEAN=[0.5]*3; rcfg.INPUT.PIXEL_STD=[0.5]*3
rcfg.DATASETS.NAMES="market1501"; rcfg.TEST.WEIGHT=W; rcfg.TEST.NECK_FEAT="before"
try:    _reid=make_model(rcfg,num_class=702,camera_num=0,view_num=1)
except Exception: _reid=make_model(rcfg,num_class=702,camera_num=6,view_num=1)
_reid.load_param(W); _reid=_reid.to(device).eval()
reid_tf=T.Compose([T.Resize([256,128]),T.ToTensor(),T.Normalize([0.5]*3,[0.5]*3)])
@torch.no_grad()
def feat_path(p):
    t=reid_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
    f=_reid(t,cam_label=None)
    if isinstance(f,(list,tuple)): f=f[0] if isinstance(f[0],torch.Tensor) else f[-1]
    if f.dim()>2: f=f.view(f.size(0),-1)
    return nn.functional.normalize(f.float(),dim=1).cpu().numpy().flatten()

# ----- Gallery: C6 real 전체 -----
print("feature 추출 (gallery=C6 전체)...")
g_f,g_p,g_c=[],[],[]
for pid in gallery_ids:
    for p in gallery_by_id[pid][GALLERY_CAM]:
        g_f.append(feat_path(p)); g_p.append(int(pid)); g_c.append(cam_idx(GALLERY_CAM))
g_f=np.array(g_f,dtype=np.float32); g_p=np.array(g_p); g_c=np.array(g_c)

# ----- Query 3종: C1 real / 생성 raw / 생성 matched -----
print("feature 추출 (query 3종)...")
def build_query(kind):
    qf,qp,qc=[],[],[]
    for pid in query_ids:
        if kind=="real":
            p=sorted(query_by_id[pid][QUERY_CAM])[0]
        elif kind=="raw":
            p=f"{GEN_DIR}/raw/{pid}_gen.png"
        else:
            p=f"{GEN_DIR}/matched/{pid}_gen.png"
        if not os.path.exists(p): continue
        qf.append(feat_path(p)); qp.append(int(pid)); qc.append(cam_idx(QUERY_CAM))
    return np.array(qf,dtype=np.float32),np.array(qp),np.array(qc)

Q_real=build_query("real"); Q_raw=build_query("raw"); Q_mat=build_query("matched")
print(f"  gallery(C6)={len(g_f)}  query real={len(Q_real[0])} raw={len(Q_raw[0])} matched={len(Q_mat[0])}\n")


# ===== 6. CP + 메트릭 =====
def build_protos(f,c): return {int(cc): f[c==cc].mean(0) for cc in np.unique(c)}
def apply_cp(f,c,protos):
    out=f.copy(); gm=np.mean(list(protos.values()),axis=0)
    for i,cc in enumerate(c): out[i]=out[i]-protos.get(int(cc),gm)
    n=np.linalg.norm(out,axis=1,keepdims=True); n[n==0]=1e-12
    return (out/n).astype(np.float32)
def cosd(qf,gf): return (1.0-qf@gf.T).astype(np.float32)
def eval_market(dm,qp,gp,qc,gc,mr=50):
    nq=dm.shape[0]; idx=np.argsort(dm,axis=1); mt=(gp[idx]==qp[:,None]).astype(np.int32)
    cmcs,APs,v=[],[],0
    for qi in range(nq):
        o=idx[qi]; keep=~((gp[o]==qp[qi])&(gc[o]==qc[qi])); raw=mt[qi][keep]
        if not raw.any(): continue
        c=raw.cumsum(); c[c>1]=1; c=c[:mr]
        if len(c)<mr: c=np.concatenate([c,np.full(mr-len(c),c[-1])])
        cmcs.append(c); v+=1; nr=raw.sum()
        tmp=raw.cumsum()/(np.arange(len(raw))+1.0); APs.append((tmp*raw).sum()/nr)
    return np.asarray(cmcs).sum(0)/v, float(np.mean(APs))

# CP proto: C6 gallery + C1 real query (real 도메인만)
protos=build_protos(np.concatenate([g_f,Q_real[0]]),
                    np.concatenate([g_c,Q_real[2]]))

def run(Q,use_cp):
    qf,qp,qc=Q
    if use_cp:
        qf2=apply_cp(qf,qc,protos); gf2=apply_cp(g_f,g_c,protos)
    else:
        qf2,gf2=qf,g_f
    cmc,mAP=eval_market(cosd(qf2,gf2),qp,g_p,qc,g_c)
    return cmc[0]*100, mAP*100

logline("\n"+"="*72)
logline(f"## [2026-06-01] script111 Query-side 포즈변환 (C1→C6) ReID")
logline(f"   Query=C1, Gallery=C6 전체({len(g_f)}장), 생성=C1외형+C6medoid포즈")
logline("="*72)
logline(f"\n{'Query 구성':<22}{'보정X R1':<10}{'보정X mAP':<11}{'CP R1':<10}{'CP mAP':<10}")
logline("-"*64)
rows=[("Baseline (C1 real)",Q_real),("Proposed (생성)",Q_raw),("Proposed+열화",Q_mat)]
bx=bc=None
for name,Q in rows:
    if len(Q[0])==0:
        logline(f"{name:<22} (생성물 없음)"); continue
    r1x,mx=run(Q,False); r1c,mc=run(Q,True)
    sx="" if bx is None else f"(mAP {mx-bx:+.1f})"
    sc="" if bc is None else f"(mAP {mc-bc:+.1f})"
    logline(f"{name:<22}{r1x:<10.1f}{mx:<11.1f}{r1c:<10.1f}{mc:<10.1f} {sx} {sc}")
    if bx is None: bx,bc=mx,mc
logline("-"*64)
logline("해석: Proposed>Baseline → query 포즈변환 양성 / +열화 추가효과 / CP 시너지")