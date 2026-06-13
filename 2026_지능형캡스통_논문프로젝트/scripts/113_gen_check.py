#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
113_gen_check.py ─ "같은 사람" 검증용 빠른 grid (5명)

적용한 개선 (조언 ①③④):
  ① IPA 입력 전 C1 원본을 512x1024 로 리사이즈 (CLIP 특징 추출 개선)
  ③ CCTV top-down full-body 프롬프트
  ④ 강화된 negative prompt
  ※ SIZE 는 384x768 유지 (은혜가 검증한 SD1.5 안정 해상도; 512x768 권고는 미적용)

비교: IPA scale 0.8 vs 1.0
출력: [C1 원본 | C6 medoid skel | scale0.8 | scale1.0]  한 줄/사람
      + CLIP-ReID cosine (생성 vs C1원본) 수치
"""
import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import torch.nn as nn
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux import OpenposeDetector

HOME=os.path.expanduser("~"); PROJECT_DIR=f"{HOME}/reid-gallery-expansion"
CACHE_DIR=f"{PROJECT_DIR}/checkpoints"; OUT=f"{PROJECT_DIR}/outputs/gen_check"
MARKET=f"/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GAL=f"{MARKET}/bounding_box_test"; QRY=f"{MARKET}/query"
device,dtype="cuda",torch.float16
SIZE=(384,768)                 # ← 유지 (검증된 안정값)
IPA_INPUT_SIZE=(512,1024)      # ← 조언①: IPA 입력용 확대
SEED=42; NUM=5; POOL=100
GALLERY_CAM="c1"; QUERY_CAM="c6"
os.makedirs(OUT,exist_ok=True)

PROMPT="a photo of a person, full body shot, standing, taken from a surveillance CCTV camera, top-down angle, highly detailed, photorealistic"
NEG="cropped, close up, partial body, missing limbs, extra limbs, bad anatomy, deformed, blurry, low resolution, face portrait"

def parse(f): p=os.path.basename(f).split("_"); return p[0],p[1][:2]

print("데이터 로드...")
gby=defaultdict(lambda:defaultdict(list))
for f in sorted(glob.glob(f"{GAL}/*.jpg")):
    pid,cam=parse(f)
    if pid in ('-1','0000'): continue
    gby[pid][cam].append(f)
qby=defaultdict(lambda:defaultdict(list))
for f in sorted(glob.glob(f"{QRY}/*.jpg")):
    pid,cam=parse(f); qby[pid][cam].append(f)
ids=[p for p in sorted(qby) if QUERY_CAM in qby[p] and GALLERY_CAM in gby[p]][:NUM]
print(f"검증 ID: {ids}\n")

print("OpenPose + C6 medoid...")
op=OpenposeDetector.from_pretrained("lllyasviel/Annotators",cache_dir=CACHE_DIR)
def skel(img):
    o=op(img)
    if isinstance(o,tuple): o=o[0]
    if o is None: return None
    if not isinstance(o,Image.Image): o=Image.fromarray(o)
    return o
def medoid(ss):
    s=np.stack([np.array(x.resize((48,96))).astype(np.float32).flatten() for x in ss])
    s=s/(np.linalg.norm(s,axis=1,keepdims=True)+1e-8); return int((s@s.T).mean(1).argmax())
idset=set(ids); pool=[]
for p in sorted(gby):
    if p in idset: continue
    if QUERY_CAM in gby[p]: pool.append(gby[p][QUERY_CAM][0])
    if len(pool)>=POOL*2: break
ss=[]
for p in pool:
    s=skel(Image.open(p).convert("RGB").resize(SIZE,Image.LANCZOS))
    if s is not None: ss.append(s)
    if len(ss)>=POOL: break
c6_skel=ss[medoid(ss)].resize(SIZE,Image.LANCZOS)
print(f"  medoid 확보 ({len(ss)}장)\n")

print("파이프라인 로드...")
cn=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose",cache_dir=CACHE_DIR,torch_dtype=dtype)
pipe=StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",controlnet=cn,cache_dir=CACHE_DIR,
    torch_dtype=dtype,safety_checker=None,requires_safety_checker=False)
pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter("h94/IP-Adapter",subfolder="models",weight_name="ip-adapter-plus_sd15.safetensors")
pipe=pipe.to(device)

def gen(c1_big, scale):
    pipe.set_ip_adapter_scale(scale)
    g=torch.Generator(device).manual_seed(SEED)
    return pipe(prompt=PROMPT,negative_prompt=NEG,image=c6_skel,ip_adapter_image=c1_big,
                controlnet_conditioning_scale=1.0,num_inference_steps=30,guidance_scale=7.5,
                width=SIZE[0],height=SIZE[1],generator=g).images[0]

print("CLIP-ReID 로드...")
sys.path.insert(0,"/home/ubuntu/CLIP-ReID")
from config import cfg as rc
from model.make_model_clipreid import make_model
W=f"{CACHE_DIR}/clipreid_duke_nosie.pth"
rc.MODEL.NAME="ViT-B-16"; rc.MODEL.STRIDE_SIZE=[16,16]; rc.MODEL.SIE_CAMERA=False
rc.MODEL.SIE_COE=0.0; rc.MODEL.ID_LOSS_TYPE="softmax"
rc.INPUT.SIZE_TRAIN=[256,128]; rc.INPUT.SIZE_TEST=[256,128]
rc.INPUT.PIXEL_MEAN=[0.5]*3; rc.INPUT.PIXEL_STD=[0.5]*3
rc.DATASETS.NAMES="market1501"; rc.TEST.WEIGHT=W; rc.TEST.NECK_FEAT="before"
try: _r=make_model(rc,num_class=702,camera_num=0,view_num=1)
except Exception: _r=make_model(rc,num_class=702,camera_num=6,view_num=1)
_r.load_param(W); _r=_r.to(device).eval()
tf=T.Compose([T.Resize([256,128]),T.ToTensor(),T.Normalize([0.5]*3,[0.5]*3)])
@torch.no_grad()
def feat(img):
    t=tf(img.convert("RGB")).unsqueeze(0).to(device)
    f=_r(t,cam_label=None)
    if isinstance(f,(list,tuple)): f=f[0] if isinstance(f[0],torch.Tensor) else f[-1]
    if f.dim()>2: f=f.view(f.size(0),-1)
    return nn.functional.normalize(f.float(),dim=1).cpu().numpy().flatten()

print("생성 + grid...\n")
cw,ch,lh=170,340,40
try: fnt=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",13)
except: fnt=ImageFont.load_default()
for pid in ids:
    c1_path=sorted(gby[pid][GALLERY_CAM])[0]
    c1_small=Image.open(c1_path).convert("RGB")
    c1_big=c1_small.resize(IPA_INPUT_SIZE,Image.LANCZOS)   # 조언①
    g08=gen(c1_big,0.8); g10=gen(c1_big,1.0)
    f_c1=feat(c1_small)
    s08=float(feat(g08)@f_c1); s10=float(feat(g10)@f_c1)
    cells=[("C1 원본",c1_small,None),("C6 medoid",c6_skel,None),
           ("scale0.8",g08,s08),("scale1.0",g10,s10)]
    grid=Image.new("RGB",(cw*len(cells)+10,ch+lh+10),"white"); d=ImageDraw.Draw(grid)
    d.text((8,4),f"PID {pid}  (생성↔C1 cosine, 높을수록 같은사람)",fill="black",font=fnt)
    for i,(l,im,sc) in enumerate(cells):
        x=i*cw+5; d.text((x,22),l,fill="navy" if sc is not None else "black",font=fnt)
        if sc is not None: d.text((x,lh-4),f"cos={sc:.3f}",fill="red",font=fnt)
        grid.paste(im.convert("RGB").resize((cw-5,ch),Image.LANCZOS),(x,lh))
    grid.save(f"{OUT}/grid_{pid}.png")
    print(f"  PID {pid}: scale0.8 cos={s08:.3f}  scale1.0 cos={s10:.3f}")
print(f"\n✅ {OUT}/grid_*.png 확인")
print("cos 높을수록 = 원본 C1 사람과 같음. 0.8 vs 1.0 중 높은 쪽 + 눈으로 자연스러운 쪽 선택")