#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
112_gallery_add_rankk.py ─ 갤러리 보강(생성본 추가) ReID, Rank-k 주지표

[은혜 설계]  데이터=Market, 모델=Duke학습본 → Duke→Market 크로스도메인
  Query (고정):  C6 real (타깃카메라 정답)
  Gallery:
    [Baseline]  C1 real 만
    [Proposed]  C1 real + 생성본(C1 인물을 C6 화각/포즈로 본뜬 것)
  변환:  IPA(외형)=C1 갤러리 인물, ControlNet(포즈)=C6 medoid skeleton
  생성본 ID = 원래 C1 사람 그대로 (정답 인정)

[지표]  Rank-1 / Rank-5 / Rank-10 (주지표, 정답개수 영향 작음) + mAP(참고)
        생성본을 정답으로 인정 → "C6 query 가 상위 k 안에 그 사람을 찾는가"
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
GEN_DIR = f"{PROJECT_DIR}/outputs/c1gallery_to_c6pose"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"
LOG = f"{PROJECT_DIR}/EXPERIMENT_LOG.md"

device, dtype = "cuda", torch.float16
GALLERY_CAM = "c1"        # 갤러리 카메라 (보강 대상)
QUERY_CAM   = "c6"        # query 카메라 (정답, 타깃 화각)
POSE_POOL_SIZE = 100
SEED, SIZE = 42, (384, 768)
IPA_SCALE, CN_SCALE = 0.8, 1.0
W = f"{CACHE_DIR}/clipreid_duke_nosie.pth"   # ★ Duke 학습본 → 크로스도메인

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
    if pid in ('-1','0000'): continue
    gallery_by_id[pid][cam].append(f)
query_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{QUERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    query_by_id[pid][cam].append(f)

# Query: C6 query 있는 사람 (정답이 C1 갤러리에 있어야 평가 가능)
query_ids = [pid for pid in sorted(query_by_id)
             if QUERY_CAM in query_by_id[pid]
             and GALLERY_CAM in gallery_by_id[pid]]
# Gallery: C1 에 등장하는 모든 사람 (distractor 포함)
gallery_ids = [pid for pid in sorted(gallery_by_id) if GALLERY_CAM in gallery_by_id[pid]]
print(f"Query(C6, 정답 C1보유): {len(query_ids)}명")
print(f"Gallery(C1 전체): {len(gallery_ids)}명\n")


# ===== 2. C6 medoid 포즈 (타깃 화각) =====
print("OpenPose + C6 medoid 포즈...")
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)
def detect_skel(img):
    try:
        o=openpose(img)
        if isinstance(o,tuple): o=o[0]
        if o is None: return None
        if not isinstance(o,Image.Image): o=Image.fromarray(o)
        return o
    except Exception: return None
def select_medoid(skels):
    s=np.stack([np.array(x.resize((48,96))).astype(np.float32).flatten() for x in skels])
    s=s/(np.linalg.norm(s,axis=1,keepdims=True)+1e-8)
    return int((s@s.T).mean(axis=1).argmax())

q_set=set(query_ids); pool=[]
for pid in sorted(gallery_by_id):
    if pid in q_set: continue
    if QUERY_CAM in gallery_by_id[pid]:        # C6 에서 자세 뽑기 (타깃 화각)
        pool.append(gallery_by_id[pid][QUERY_CAM][0])
    if len(pool)>=POSE_POOL_SIZE*2: break
skels=[]
for p in pool:
    sk=detect_skel(Image.open(p).convert("RGB").resize(SIZE,Image.LANCZOS))
    if sk is not None: skels.append(sk)
    if len(skels)>=POSE_POOL_SIZE: break
c6_medoid_skel=skels[select_medoid(skels)].resize(SIZE,Image.LANCZOS)
print(f"  C6 medoid pose 확보 ({len(skels)}장)\n")


# ===== 3. 열화 =====
def analyze_cctv(ref):
    a=np.asarray(ref.convert("RGB")).astype(np.float32); gray=a.mean(2)
    lap=np.abs(np.gradient(np.gradient(gray,axis=0)[0],axis=0)).var()
    bl=Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
    return dict(sharp=float(lap),noise=float((gray-np.asarray(bl)).std()),
                bright=float(gray.mean()),contrast=float(gray.std()),
                sat=float(np.asarray(ref.convert("HSV")).astype(np.float32)[:,:,1].mean()))
def degrade_matched(img,ref):
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


# ===== 4. 생성: C1 갤러리 인물 → C6 화각 =====
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
print("로드 완료. 생성 시작 (C1 갤러리 인물 → C6 화각)...")

# 갤러리 모든 사람의 C1 대표 이미지를 C6 화각으로 변환
for i, pid in enumerate(gallery_ids):
    # 변경할 코드 (IP-Adapter가 특징을 잘 잡을 수 있게 크기 키우기)
    raw_c1_img = Image.open(sorted(gallery_by_id[pid][GALLERY_CAM])[0]).convert("RGB")
    # LANCZOS 필터로 512x1024 정도로 부드럽게 키워서 IP-Adapter에 입력
    c1_img = raw_c1_img.resize((512, 1024), Image.LANCZOS)
    raw_p=f"{GEN_DIR}/raw/{pid}_gen.png"
    mat_p=f"{GEN_DIR}/matched/{pid}_gen.png"
    if os.path.exists(raw_p) and os.path.exists(mat_p): continue
    g=torch.Generator(device).manual_seed(SEED)
    gen=pipe(prompt="a photo of a person, full body, surveillance",
             negative_prompt="blurry, low quality, deformed, multiple people, extra limbs",
             image=c6_medoid_skel,        # 포즈 = C6 화각
             ip_adapter_image=c1_img,     # 외형 = C1 갤러리 인물
             controlnet_conditioning_scale=CN_SCALE,
             num_inference_steps=30, guidance_scale=7.5,
             width=SIZE[0], height=SIZE[1], generator=g).images[0]
    gen.save(raw_p); degrade_matched(gen,c1_img).save(mat_p)
    if (i+1)%50==0: print(f"  {i+1}/{len(gallery_ids)}")
del pipe,cn; gc.collect(); torch.cuda.empty_cache()
print("생성 완료\n")


# ===== 5. CLIP-ReID (Duke 학습본) =====
print("CLIP-ReID 로드 (Duke 학습본 → 크로스도메인)...")
sys.path.insert(0,"/home/ubuntu/CLIP-ReID")
from config import cfg as rcfg
from model.make_model_clipreid import make_model
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

# Query: C6 real
print("feature 추출...")
q_f,q_p,q_c=[],[],[]
for pid in query_ids:
    for p in query_by_id[pid][QUERY_CAM]:
        q_f.append(feat_path(p)); q_p.append(int(pid)); q_c.append(cam_idx(QUERY_CAM))
# Gallery base: C1 real
g_f,g_p,g_c=[],[],[]
for pid in gallery_ids:
    for p in gallery_by_id[pid][GALLERY_CAM]:
        g_f.append(feat_path(p)); g_p.append(int(pid)); g_c.append(cam_idx(GALLERY_CAM))
# 생성본 (C1 인물 → C6 화각), 갤러리에 추가될 후보. cam = c1 소속(원래 사람의 카메라)
raw_f,raw_p,raw_c=[],[],[]; mat_f,mat_p,mat_c=[],[],[]
for pid in gallery_ids:
    rp=f"{GEN_DIR}/raw/{pid}_gen.png"; mp=f"{GEN_DIR}/matched/{pid}_gen.png"
    if os.path.exists(rp):
        raw_f.append(feat_path(rp)); raw_p.append(int(pid)); raw_c.append(cam_idx(GALLERY_CAM))
    if os.path.exists(mp):
        mat_f.append(feat_path(mp)); mat_p.append(int(pid)); mat_c.append(cam_idx(GALLERY_CAM))
A=lambda x:np.array(x,dtype=np.float32)
q_f,g_f,raw_f,mat_f=map(A,[q_f,g_f,raw_f,mat_f])
q_p,q_c,g_p,g_c=map(np.array,[q_p,q_c,g_p,g_c])
raw_p,raw_c,mat_p,mat_c=map(np.array,[raw_p,raw_c,mat_p,mat_c])
print(f"  query(C6)={len(q_f)} gallery(C1)={len(g_f)} gen_raw={len(raw_f)} gen_matched={len(mat_f)}\n")


# ===== 6. Rank-k + mAP =====
def cosd(qf,gf): return (1.0-qf@gf.T).astype(np.float32)
def eval_ranks(dm,qp,gp,qc,gc,topk=(1,5,10),mr=50):
    nq=dm.shape[0]; idx=np.argsort(dm,axis=1); mt=(gp[idx]==qp[:,None]).astype(np.int32)
    cmcs,APs,v=[],[],0
    for qi in range(nq):
        o=idx[qi]; keep=~((gp[o]==qp[qi])&(gc[o]==qc[qi])); raw=mt[qi][keep]
        if not raw.any(): continue
        c=raw.cumsum(); c[c>1]=1; c=c[:mr]
        if len(c)<mr: c=np.concatenate([c,np.full(mr-len(c),c[-1])])
        cmcs.append(c); v+=1; nr=raw.sum()
        tmp=raw.cumsum()/(np.arange(len(raw))+1.0); APs.append((tmp*raw).sum()/nr)
    cmc=np.asarray(cmcs).sum(0)/v
    return {f"R{k}":float(cmc[k-1]*100) for k in topk}, float(np.mean(APs))*100

# ----- CP (카메라 프로토타입 보정) -----
def build_protos(f,c): return {int(cc): f[c==cc].mean(0) for cc in np.unique(c)}
def apply_cp(f,c,protos):
    out=f.copy(); gm=np.mean(list(protos.values()),axis=0)
    for i,cc in enumerate(c): out[i]=out[i]-protos.get(int(cc),gm)
    n=np.linalg.norm(out,axis=1,keepdims=True); n[n==0]=1e-12
    return (out/n).astype(np.float32)

# proto: real 도메인만 (C6 query + C1 gallery). 생성본은 proto 산정에서 제외(누출 방지)
protos=build_protos(np.concatenate([q_f,g_f]), np.concatenate([q_c,g_c]))

def run(extra_f,extra_p,extra_c,use_cp):
    if extra_f is None or len(extra_f)==0:
        G_f,G_p,G_c=g_f,g_p,g_c
    else:
        G_f=np.concatenate([g_f,extra_f]); G_p=np.concatenate([g_p,extra_p]); G_c=np.concatenate([g_c,extra_c])
    if use_cp:
        qf=apply_cp(q_f,q_c,protos); gf=apply_cp(G_f,G_c,protos)
    else:
        qf,gf=q_f,G_f
    return eval_ranks(cosd(qf,gf),q_p,G_p,q_c,G_c)

logline("\n"+"="*72)
logline(f"## [2026-06-03] script112 갤러리 보강 + CP (C1+생성→C6화각) Duke→Market 크로스도메인")
logline(f"   Query=C6 real({len(q_f)}), Gallery=C1 real({len(g_f)})+생성본, Rank-k 주지표")
logline("="*72)
logline(f"\n{'구성':<18}{'보정X R1':<9}{'보정X R5':<9}{'보정X mAP':<11}{'CP R1':<8}{'CP R5':<8}{'CP mAP':<9}")
logline("-"*72)
rows=[("Baseline (C1)",None,None,None),
      ("+ gen_raw",raw_f,raw_p,raw_c),
      ("+ gen_matched",mat_f,mat_p,mat_c)]
b_r1x=b_mx=b_r1c=b_mc=None
for name,ef,ep,ec in rows:
    rx,mx=run(ef,ep,ec,False)
    rc,mc=run(ef,ep,ec,True)
    if b_r1x is None:
        sx=sc=""
        b_r1x,b_mx,b_r1c,b_mc=rx['R1'],mx,rc['R1'],mc
    else:
        sx=f"(R1{rx['R1']-b_r1x:+.1f})"; sc=f"(R1{rc['R1']-b_r1c:+.1f})"
    logline(f"{name:<18}{rx['R1']:<9.1f}{rx['R5']:<9.1f}{mx:<11.1f}{rc['R1']:<8.1f}{rc['R5']:<8.1f}{mc:<9.1f} {sx}{sc}")
logline("-"*72)
logline("해석: 세로=생성보강효과 / 가로(보정X→CP)=카메라보정효과 / +gen+CP 조합이 최고면 둘 다 독립 기여")