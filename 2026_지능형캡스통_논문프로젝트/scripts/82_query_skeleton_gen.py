"""
82_query_skeleton_gen.py

핵심 변경: 이전(타겟 카메라 아무 자세) → 지금(query 자체 자세로 합성)

파이프라인:
  1. 보정(global mean) + baseline 매칭 → Top-K 후보 + confidence
  2. confidence 낮은 query 선별 (하위 ratio)
  3. 선별된 query의 skeleton 추출
  4. Top-K 후보 c1 인물을 query skeleton으로 합성 (IP-Adapter 외형 + ControlNet 자세)
  5. 합성물 vs query 재매칭 → 점수결합

query skeleton 사용 = leakage 아님 (추론 시 query는 입력)
SIE 없는 weight. 보정(global). 

먼저 c5 테스트 (confidence 하위 20%, 5명 → 5×TopK=25장 생성).
비교 이미지도 저장: [c1원본 | query | query_skel | 합성물]
"""
import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import random

sys.path.insert(0, "/home/ubuntu/CLIP-ReID")
from config import cfg
from model.make_model_clipreid import make_model

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CKPT = f"{PROJECT_DIR}/checkpoints"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
MARKET_GALLERY = f"{MARKET_DIR}/bounding_box_test"
MARKET_QUERY = f"{MARKET_DIR}/query"
OUT_DIR = f"{PROJECT_DIR}/outputs/queryskel_gen"
OUT_COMPARE = f"{PROJECT_DIR}/outputs/queryskel_compare"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_COMPARE, exist_ok=True)

device="cuda"; SIZE=(384,768); random.seed(42)
TARGET_CAMS = ["c5"]
NUM_IDS = 100; TOP_K = 5; ALPHA = 0.7; N_GLOBAL = 50

def parse(f):
    p=os.path.basename(f).split("_"); return p[0],p[1][:2]

# ===== Re-ID model (SIE 없음) =====
def load_nosie(wp, ds, nc, cn):
    cfg.MODEL.NAME='ViT-B-16'; cfg.MODEL.STRIDE_SIZE=[16,16]
    cfg.MODEL.SIE_CAMERA=False; cfg.MODEL.SIE_COE=0.0; cfg.MODEL.ID_LOSS_TYPE='softmax'
    cfg.INPUT.SIZE_TRAIN=[256,128]; cfg.INPUT.SIZE_TEST=[256,128]
    cfg.INPUT.PIXEL_MEAN=[0.5,0.5,0.5]; cfg.INPUT.PIXEL_STD=[0.5,0.5,0.5]
    cfg.DATASETS.NAMES=ds; cfg.TEST.WEIGHT=wp; cfg.TEST.NECK_FEAT='before'
    try: m=make_model(cfg,num_class=nc,camera_num=0,view_num=1)
    except: m=make_model(cfg,num_class=nc,camera_num=cn,view_num=1)
    m.load_param(wp); return m.eval().to(device)

reid_transform = T.Compose([T.Resize([256,128]),T.ToTensor(),
                            T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
@torch.no_grad()
def feat_raw(model,path_or_img):
    if isinstance(path_or_img,str):
        img=Image.open(path_or_img).convert("RGB")
    else:
        img=path_or_img
    t=reid_transform(img).unsqueeze(0).to(device)
    try: f=model(t)
    except TypeError: f=model(t,cam_label=None)
    return f.cpu().numpy().flatten()
def l2n(x): return x/(np.linalg.norm(x)+1e-9)

# ===== 생성 모듈 =====
def normalize_skeleton(skel_img, target_h_ratio=0.85):
    arr=np.array(skel_img); mask=arr.sum(axis=2)>20; ys,xs=np.where(mask)
    if len(ys)==0: return skel_img
    y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
    crop=arr[y0:y1+1,x0:x1+1]; ch,cw=crop.shape[:2]; W,H=SIZE
    target_h=int(H*target_h_ratio); scale=target_h/ch
    new_w=max(1,int(cw*scale)); new_h=target_h
    crop_img=Image.fromarray(crop).resize((new_w,new_h),Image.LANCZOS)
    canvas=Image.new("RGB",SIZE,(0,0,0))
    px=(W-new_w)//2; py=(H-new_h)//2
    canvas.paste(crop_img,(px,py)); return canvas

def build_gen_pipe():
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
    from controlnet_aux import OpenposeDetector
    cn=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose",
                                        cache_dir=CKPT,torch_dtype=torch.float16)
    pipe=StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",controlnet=cn,
        cache_dir=CKPT,torch_dtype=torch.float16,safety_checker=None,requires_safety_checker=False)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    pipe=pipe.to(device)
    pipe.load_ip_adapter("h94/IP-Adapter",subfolder="models",
                         weight_name="ip-adapter-plus_sd15.safetensors",cache_dir=CKPT)
    pipe.set_ip_adapter_scale(1.0)
    # enable_attention_slicing 제거 (IP-Adapter 충돌)
    op=OpenposeDetector.from_pretrained("lllyasviel/Annotators",cache_dir=CKPT)
    return pipe,op

@torch.no_grad()
def gen_with_query_skel(pipe, openpose, c1_path, query_path, seed=42):
    """c1 외형(IP-Adapter) + query 자세(ControlNet) → 합성"""
    c1_img=Image.open(c1_path).convert("RGB").resize(SIZE,Image.LANCZOS)
    q_img=Image.open(query_path).convert("RGB").resize(SIZE,Image.LANCZOS)
    skel=openpose(q_img)   # ★ query의 자세 추출
    skel=normalize_skeleton(skel.resize(SIZE,Image.LANCZOS))
    emb=pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=c1_img,ip_adapter_image_embeds=None,
        device=device,num_images_per_prompt=1,do_classifier_free_guidance=True)
    result=pipe(
        prompt="a photo of one single person, solo, full body, standing, centered, surveillance camera",
        negative_prompt="two people, multiple people, group, crowd, duplicate, blurry, low quality, deformed, cropped",
        ip_adapter_image_embeds=emb, image=skel,
        controlnet_conditioning_scale=1.0, num_inference_steps=30, guidance_scale=7.5,
        generator=torch.Generator(device=device).manual_seed(seed),
        width=SIZE[0],height=SIZE[1],
    ).images[0]
    return result, skel

def make_compare(c1_path, q_path, skel, gen, save_path):
    """[c1원본 | query | skeleton | 합성] 가로 4장"""
    c1=Image.open(c1_path).convert("RGB").resize(SIZE,Image.LANCZOS)
    q=Image.open(q_path).convert("RGB").resize(SIZE,Image.LANCZOS)
    W,H=SIZE; gap=10
    canvas=Image.new("RGB",(W*4+gap*3,H),(255,255,255))
    canvas.paste(c1,(0,0))
    canvas.paste(q,(W+gap,0))
    canvas.paste(skel.resize(SIZE),(W*2+gap*2,0))
    canvas.paste(gen.resize(SIZE),(W*3+gap*3,0))
    canvas.save(save_path)

# ===== 메인 =====
def main():
    gby=defaultdict(lambda:defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_GALLERY}/*.jpg")):
        pid,cam=parse(f)
        if pid in ('-1','0000'): continue
        gby[pid][cam].append(f)
    qby=defaultdict(lambda:defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_QUERY}/*.jpg")):
        pid,cam=parse(f); qby[pid][cam].append(f)
    all_qf=[f for f in sorted(glob.glob(f"{MARKET_QUERY}/*.jpg")) if parse(f)[0] not in ('-1','0000')]

    for tc in TARGET_CAMS:
        ids=[]
        for pid in sorted(gby.keys()):
            if "c1" in gby[pid] and tc in qby[pid]: ids.append(pid)
            if len(ids)>=NUM_IDS: break

        # === Phase 1: Re-ID feature 추출 + baseline + confidence ===
        print(f"\n=== {tc}: Re-ID feature 추출 ===")
        reid = load_nosie(f"{CKPT}/clipreid_duke_nosie.pth","dukemtmcreid",702,8)
        c1_raw,q_raw,kept=[],[],[]
        c1_paths,q_paths=[],[]
        for pid in ids:
            c1_paths.append(sorted(gby[pid]["c1"])[0])
            q_paths.append(sorted(qby[pid][tc])[0])
            c1_raw.append(feat_raw(reid,c1_paths[-1]))
            q_raw.append(feat_raw(reid,q_paths[-1]))
            kept.append(pid)
        c1_raw=np.array(c1_raw); q_raw=np.array(q_raw); N=len(kept)
        # global mean (평가 query 제외)
        eval_q=set(q_paths)
        pool=[f for f in all_qf if f not in eval_q]
        random.Random(42).shuffle(pool)
        gmean=np.mean([feat_raw(reid,f) for f in pool[:N_GLOBAL]],axis=0)
        # 보정
        qf=np.array([l2n(x-gmean) for x in q_raw])
        c1f=np.array([l2n(x-gmean) for x in c1_raw])
        sims=qf@c1f.T
        # confidence = top1-top2 margin
        margins=[]
        for i in range(N):
            s=np.sort(sims[i])[::-1]
            margins.append(s[0]-s[1])
        margins=np.array(margins)
        base_pred=sims.argmax(axis=1)
        base_correct=(base_pred==np.arange(N))
        B_r1=base_correct.mean()*100
        print(f"  보정 baseline R1 = {B_r1:.2f}%")

        # confidence 하위 20% 선별
        ratio=0.2
        thr=np.quantile(margins,ratio)
        sel_idx=np.where(margins<=thr)[0]
        print(f"  confidence 하위 {ratio:.0%}: {len(sel_idx)}명 선별 (threshold={thr:.4f})")
        print(f"  이 중 baseline 틀린 수: {(~base_correct[sel_idx]).sum()}")

        # === Phase 2: 선별된 query만 생성 ===
        del reid; torch.cuda.empty_cache()
        print(f"\n=== {tc}: 생성 (query skeleton 기반, {len(sel_idx)}명 × Top-{TOP_K}) ===")
        pipe,openpose=build_gen_pipe()
        os.makedirs(f"{OUT_DIR}/{tc}",exist_ok=True)
        os.makedirs(f"{OUT_COMPARE}/{tc}",exist_ok=True)
        gen_results={}  # sel_i → {rank_j: gen_path}
        for si in sel_idx:
            pid=kept[si]; q_path=q_paths[si]
            topk=np.argsort(-sims[si])[:TOP_K]
            gen_paths=[]
            for j,cand in enumerate(topk):
                cand_c1=c1_paths[cand]
                save=f"{OUT_DIR}/{tc}/{pid}_q_cand{j}_{kept[cand]}.png"
                gen_img,skel=gen_with_query_skel(pipe,openpose,cand_c1,q_path,seed=42+j)
                gen_img.save(save)
                gen_paths.append(save)
                # 첫 후보만 비교 저장
                if j==0:
                    make_compare(cand_c1,q_path,skel,gen_img,
                                 f"{OUT_COMPARE}/{tc}/{pid}_compare.png")
            gen_results[si]=gen_paths
        del pipe; torch.cuda.empty_cache()

        # === Phase 3: 재매칭 ===
        print(f"\n=== {tc}: 재매칭 ===")
        reid = load_nosie(f"{CKPT}/clipreid_duke_nosie.pth","dukemtmcreid",702,8)
        correct_after=0
        recover=0; damage=0
        for i in range(N):
            if i in gen_results:
                # 선별된 query: 재매칭
                topk=np.argsort(-sims[i])[:TOP_K]
                gen_feats=[]
                for gp in gen_results[i]:
                    gf=l2n(feat_raw(reid,gp)-gmean)
                    gen_feats.append(gf)
                gen_feats=np.array(gen_feats)
                s2=qf[i]@gen_feats.T
                s1_topk=sims[i][topk]
                final=ALPHA*s1_topk+(1-ALPHA)*s2
                best=topk[final.argmax()]
                got_it=(best==i)
                if got_it: correct_after+=1
                if got_it and not base_correct[i]: recover+=1
                if not got_it and base_correct[i]: damage+=1
            else:
                # 비선별: baseline 유지
                if base_correct[i]: correct_after+=1
        after_r1=correct_after/N*100
        print(f"\n결과:")
        print(f"  보정 baseline: {B_r1:.2f}%")
        print(f"  +선별생성재매칭: {after_r1:.2f}% ({after_r1-B_r1:+.2f})")
        print(f"  회복: {recover}  손상: {damage}  순: {recover-damage:+d}")
        print(f"  선별 {len(sel_idx)}명 중 baseline 틀린 {(~base_correct[sel_idx]).sum()}명")
        print(f"\n비교이미지: {OUT_COMPARE}/{tc}/")
        print("  [c1원본 | query | query_skeleton | 합성물]")
        del reid; torch.cuda.empty_cache()

if __name__=="__main__":
    main()