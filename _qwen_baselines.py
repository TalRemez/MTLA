import torch, glob, json, numpy as np
from mtla.mtla_attn import reduce_band
from mtla.voting import vote
from mtla.utils import overlap_fn
from mtla.metrics import coco_map
GT='data/coco/annotations/instances_val2017.json'
gt=json.load(open(GT))
name2cat={c['name'].lower():c['id'] for c in gt['categories']}
img_wh={im['id']:(im['width'],im['height']) for im in gt['images']}
rolls=sorted(int(d.split('rollout')[1]) for d in glob.glob('runs/coco/qwen3vl_image/features/rollout*'))
print('rollouts:', rolls, flush=True)
recs_by_roll={}
for rk in rolls:
    rr=[]
    for f in glob.glob(f'runs/coco/qwen3vl_image/features/rollout{rk}/shard*.pt'):
        rr+=torch.load(f,weights_only=False)
    recs_by_roll[rk]=rr
def build_cands(sig):
    cands=[]
    for rk in rolls:
        for rec in recs_by_roll[rk]:
            for o in rec['objects']:
                score=1.0 if sig=='raw' else (float(reduce_band(dict(o)[sig].astype(np.float32))) if dict(o).get(sig) is not None else 0.0)
                cands.append({'id':rec['id'],'label':o['label'],'region':o['region'],'score':score,'hallu':bool(o['is_hallucinated']),'extracted':True,'rollout':rk})
    return cands
def ap_for(sig, agg):
    voted=vote(build_cands(sig), agg=agg, iou_fn=overlap_fn('iou'), top_k=None)
    dets=[]
    for (iid,label),kept in voted.items():
        cid=name2cat.get((label or '').strip().lower())
        if cid is None: continue
        W,H=img_wh.get(iid,(1,1))
        for (x1,y1,x2,y2),sc in kept:
            dets.append({'image_id':iid,'category_id':cid,'bbox':[x1/1000*W,y1/1000*H,(x2-x1)/1000*W,(y2-y1)/1000*H],'score':float(sc)})
    return coco_map(dets, GT), len(dets)
print(f'{"baseline":<16s}{"agg":>8s}{"mAP":>8s}{"mAP50":>8s}{"mAP75":>8s}{"ndets":>10s}', flush=True)
for sig,agg in [('raw','sum'),('first_global','support'),('digits_local','support'),('all_local','support'),('all_global','support')]:
    m,nd=ap_for(sig,agg)
    print(f'{sig:<16s}{agg:>8s}{m["mAP"]:>8.2f}{m["mAP50"]:>8.2f}{m["mAP75"]:>8.2f}{nd:>10d}', flush=True)
