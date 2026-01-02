from __future__ import annotations
import random, math, time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# -----------------------------
# World and shapes
# -----------------------------
H, W = 12, 32
N = H * W
HORIZONS = [1, 2, 4, 8, 16, 32, 64]

OCC_X0, OCC_X1 = 12, 19  # occluder band
OCC_MASK = [0]*N
for y in range(H):
    for x in range(OCC_X0, OCC_X1+1):
        OCC_MASK[y*W + x] = 1
OCC_IDXS = [i for i,v in enumerate(OCC_MASK) if v==1]

SHAPES = {
    "square": [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)],
    "cross":  [(0,0),(-1,0),(1,0),(0,-1),(0,1)],
    "ell":    [(0,0),(0,1),(0,2),(1,2),(2,2)],
    "line_h": [(-2,0),(-1,0),(0,0),(1,0),(2,0)],
    "line_v": [(0,-2),(0,-1),(0,0),(0,1),(0,2)],
}

def idx(x:int,y:int)->int: return y*W + x

def render_shape(name:str, cx:int, cy:int)->List[int]:
    g = [0]*N
    for dx,dy in SHAPES[name]:
        x,y = cx+dx, cy+dy
        if 0<=x<W and 0<=y<H:
            g[idx(x,y)] = 1
    return g

def mean(xs: List[float])->float: return sum(xs)/max(1,len(xs))
def std(xs: List[float])->float:
    if len(xs)<=1: return 0.0
    m=mean(xs); return math.sqrt(mean([(x-m)**2 for x in xs]))

def truth_in_occluder(truth: List[int]) -> bool:
    return any(truth[i]==1 for i in OCC_IDXS)

def iou_occ(pred_occ: List[float], truth: List[int], thr: float=0.5)->float:
    inter=0; union=0
    for j,i in enumerate(OCC_IDXS):
        p = 1 if pred_occ[j]>=thr else 0
        g = truth[i]
        if p==1 and g==1: inter += 1
        if p==1 or g==1: union += 1
    return 1.0 if union==0 else inter/union

def centroid(truth: List[int], idxs: List[int]) -> Optional[Tuple[float,float]]:
    pts=[]
    for i in idxs:
        if truth[i]==1:
            y=i//W; x=i%W
            pts.append((x,y))
    if not pts: return None
    return (mean([p[0] for p in pts]), mean([p[1] for p in pts]))

def centroid_pred(pred_occ: List[float], thr: float=0.5) -> Optional[Tuple[float,float]]:
    pts=[]
    for j,i in enumerate(OCC_IDXS):
        if pred_occ[j]>=thr:
            y=i//W; x=i%W
            pts.append((x,y))
    if not pts: return None
    return (mean([p[0] for p in pts]), mean([p[1] for p in pts]))

def l2(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

@dataclass
class Episode:
    h: int
    shape: str
    cx: int
    cy: int
    vx: int
    vy: int
    t_start: int
    t_occl: int
    t_reveal: int
    t_end: int

class ChamberWorld:
    """
    Non-degenerate: when occluded, object bounces INSIDE the occluder columns.
    At reveal tick, occluder turns off but object is still inside the band.
    """
    def __init__(self, seed:int=0, pre:int=6, post:int=2):
        self.rng=random.Random(seed)
        self.t=0
        self.pre=pre
        self.post=post
        self.ep: Optional[Episode]=None
        self._new_ep()

    def _new_ep(self):
        h=self.rng.choice(HORIZONS)
        shape=self.rng.choice(list(SHAPES.keys()))
        cy=self.rng.randint(3, H-4)
        cx=OCC_X0-3
        vx=1
        vy=self.rng.choice([-1,0,1])
        t0=self.t
        t_occl=t0+self.pre
        t_reveal=t_occl+h
        t_end=t_reveal+self.post
        self.ep=Episode(h,shape,cx,cy,vx,vy,t0,t_occl,t_reveal,t_end)

    def step(self)->Tuple[List[int],List[int],bool,bool,bool,int]:
        assert self.ep is not None
        ep=self.ep
        is_occ=(ep.t_occl<=self.t<ep.t_reveal)
        is_start=(self.t==ep.t_occl)
        is_reveal=(self.t==ep.t_reveal)

        # move
        ep.cx += ep.vx
        ep.cy += ep.vy
        if is_occ:
            if ep.cx < OCC_X0+1 or ep.cx > OCC_X1-1:
                ep.vx *= -1
                ep.cx = max(OCC_X0+1, min(OCC_X1-1, ep.cx))
        if ep.cy < 2 or ep.cy > H-3:
            ep.vy *= -1
            ep.cy = max(2, min(H-3, ep.cy))

        truth = render_shape(ep.shape, ep.cx, ep.cy)
        occ = OCC_MASK[:] if is_occ else [0]*N

        self.t += 1
        if self.t >= ep.t_end:
            self._new_ep()

        return truth, occ, is_occ, is_start, is_reveal, ep.h

    def observe(self, truth: List[int], occ: List[int]) -> Tuple[List[int],List[int]]:
        # FULL FIELD except occluded cells are UNOBSERVED (missing)
        vals=[]; idxs=[]
        for i in range(N):
            if occ[i]==1: continue
            idxs.append(i); vals.append(truth[i])
        return vals, idxs

# -----------------------------
# Multi-table SimHash + banded buckets
# Key fix: use 8-bit bands to avoid “no candidates” at small library sizes.
# 64 bits => 8 bands => 8-bit per band.
# -----------------------------

class MultiSimHashBands:
    def __init__(self, feat_len:int, n_tables:int=5, bands_per_table:int=8, seed:int=12345):
        assert 64 % bands_per_table == 0
        self.n_tables=n_tables
        self.bands=bands_per_table
        self.band_bits=64//bands_per_table  # here = 8
        self.weights=[]
        for t in range(n_tables):
            rng=random.Random(seed+100000*t)
            self.weights.append([[1 if rng.getrandbits(1) else -1 for _ in range(feat_len)] for _ in range(64)])

    def sig(self, v: List[int]) -> Tuple[int,...]:
        out=[]
        for t in range(self.n_tables):
            bits=0
            wt=self.weights[t]
            for i in range(64):
                s=0
                wi=wt[i]
                for k,x in enumerate(v):
                    s += wi[k]*x
                if s>=0: bits |= (1<<i)
            out.append(bits)
        return tuple(out)

    def bucket_keys(self, sig: Tuple[int,...]) -> List[Tuple[int,int,int]]:
        keys=[]
        mask=(1<<self.band_bits)-1  # 0..255
        for t in range(self.n_tables):
            s=sig[t]
            for b in range(self.bands):
                val=(s>>(b*self.band_bits)) & mask
                keys.append((t,b,val))
        return keys

def sig_dist(a: Tuple[int,...], b: Tuple[int,...]) -> int:
    return sum((a[i]^b[i]).bit_count() for i in range(len(a)))

class BandIndex:
    def __init__(self, scheme: MultiSimHashBands, bucket_cap:int=16, cand_max:int=256):
        self.scheme=scheme
        self.bucket_cap=bucket_cap
        self.cand_max=cand_max
        self.buckets: Dict[Tuple[int,int,int,int], List[int]] = {}

    def insert(self, h:int, uid:int, sig:Tuple[int,...]) -> None:
        for (t,b,val) in self.scheme.bucket_keys(sig):
            k=(h,t,b,val)
            arr=self.buckets.get(k)
            if arr is None:
                arr=[]; self.buckets[k]=arr
            if len(arr)>=self.bucket_cap:
                arr.pop(0)
            arr.append(uid)

    def candidates(self, h:int, sig_q:Tuple[int,...]) -> List[int]:
        cand=[]
        seen={}
        for (t,b,val) in self.scheme.bucket_keys(sig_q):
            arr=self.buckets.get((h,t,b,val), [])
            for uid in arr:
                if uid in seen: continue
                seen[uid]=True
                cand.append(uid)
                if len(cand)>=self.cand_max:
                    return cand
        return cand

# -----------------------------
# NUPCA-like memory agent + FullScan oracle variant
# -----------------------------

@dataclass
class Unit:
    uid: int
    h: int
    sig: Tuple[int,...]
    pred_occ: List[float]
    err_ema: float = 1.0
    val_count: int = 0

@dataclass
class Pending:
    t_due: int
    h: int
    uid: int
    pred_snapshot: List[float]

class LegacyBandAgent:
    def __init__(self, full_scan: bool, seed:int=0):
        self.full_scan=full_scan
        self.rng=random.Random(seed)
        self.prev_row=[0]*H
        self.prev_col=[0]*W
        self.feat_len=(H+W)*2 + len(HORIZONS) + 3
        self.scheme=MultiSimHashBands(self.feat_len, n_tables=5, bands_per_table=8, seed=seed+123)
        self.index=BandIndex(self.scheme, bucket_cap=16, cand_max=256)

        self.mem={h: [] for h in HORIZONS}
        self.pending: Dict[int,List[Pending]] = {}
        self.belief_occ=[0.0]*len(OCC_IDXS)
        self.t=0

        # creation gates
        self.min_new_dist=100
        self.persist_needed=1
        self.novelty_ctr={h:0 for h in HORIZONS}

        # retrieval params
        self.K=7
        self.alpha_err=2.0

        # logs
        self.stage1_hits=0
        self.stage1_queries=0
        self.stage1_cand_sum=0
        self.stage2_scored_sum=0
        self.created={h:0 for h in HORIZONS}
        self.validated={h:0 for h in HORIZONS}

    def _shape_bits(self, obs_vals:List[int], obs_idx:List[int]) -> List[int]:
        left=mid=right=0
        for v,i in zip(obs_vals, obs_idx):
            if v==0: continue
            x=i%W
            if x < OCC_X0: left += 1
            elif x > OCC_X1: right += 1
            else: mid += 1
        return [1 if left>0 else 0, 1 if right>0 else 0, 1 if mid>0 else 0]

    def _features(self, obs_vals:List[int], obs_idx:List[int], h:int) -> List[int]:
        row=[0]*H
        col=[0]*W
        for v,i in zip(obs_vals, obs_idx):
            if v==0: continue
            y=i//W; x=i%W
            row[y]+=1; col[x]+=1
        drow=[row[i]-self.prev_row[i] for i in range(H)]
        dcol=[col[i]-self.prev_col[i] for i in range(W)]
        self.prev_row=row; self.prev_col=col
        feats=row+col+drow+dcol
        for hh in HORIZONS: feats.append(1 if hh==h else 0)
        feats += self._shape_bits(obs_vals, obs_idx)
        return feats

    def _retrieve(self, h:int, sig_q:Tuple[int,...]) -> List[Unit]:
        units=self.mem[h]
        self.stage1_queries += 1
        if not units:
            return []
        if self.full_scan:
            cand = list(range(len(units)))
        else:
            cand = self.index.candidates(h, sig_q)

        if cand:
            self.stage1_hits += 1
        self.stage1_cand_sum += len(cand)

        scored=[]
        for uid in cand:
            self.stage2_scored_sum += 1
            u=units[uid]
            d=sig_dist(sig_q, u.sig)
            score=float(d) + self.alpha_err*u.err_ema
            scored.append((score, uid))
        scored.sort(key=lambda x:(x[0], x[1]))
        return [units[uid] for _,uid in scored[:self.K]]

    def on_occlusion_start(self, obs_vals:List[int], obs_idx:List[int], h_occ:int) -> None:
        feats=self._features(obs_vals, obs_idx, h_occ)
        sig_q=self.scheme.sig(feats)
        retrieved=self._retrieve(h_occ, sig_q)

        pred=self.belief_occ[:]
        if retrieved:
            num=[0.0]*len(OCC_IDXS)
            den=[0.0]*len(OCC_IDXS)
            for u in retrieved:
                w=0.2 if u.val_count==0 else 1.0/max(0.05,u.err_ema)
                for j in range(len(OCC_IDXS)):
                    num[j]+=w*u.pred_occ[j]; den[j]+=w
            for j in range(len(OCC_IDXS)):
                if den[j]>0: pred[j]=num[j]/den[j]

        # compare-first creation (scan distance is ok because mem is bounded in this toy)
        best_d=10**9
        for u in self.mem[h_occ]:
            best_d=min(best_d, sig_dist(sig_q, u.sig))
        if best_d>self.min_new_dist:
            self.novelty_ctr[h_occ]+=1
        else:
            self.novelty_ctr[h_occ]=0
        if (best_d>self.min_new_dist) and (self.novelty_ctr[h_occ]>=self.persist_needed):
            uid=len(self.mem[h_occ])
            self.mem[h_occ].append(Unit(uid,h_occ,sig_q,pred[:],1.0,0))
            if not self.full_scan:
                self.index.insert(h_occ, uid, sig_q)
            self.created[h_occ]+=1

        # validate best matching unit if exists else skip
        use_uid = len(self.mem[h_occ])-1 if self.mem[h_occ] else -1
        self.pending.setdefault(self.t+h_occ, []).append(Pending(self.t+h_occ, h_occ, use_uid, pred[:]))
        self.belief_occ = pred[:]

    def tick(self, obs_vals:List[int], obs_idx:List[int], is_occluded:bool, is_reveal_tick:bool) -> None:
        # validate due
        due=self.pending.pop(self.t, [])
        if due and is_reveal_tick:
            obs_map={i:float(v) for i,v in zip(obs_idx, obs_vals)}
            idx_to_j={OCC_IDXS[j]:j for j in range(len(OCC_IDXS))}
            for p in due:
                if p.uid<0 or p.uid>=len(self.mem[p.h]): continue
                u=self.mem[p.h][p.uid]
                err=mean([abs(p.pred_snapshot[idx_to_j[i]] - obs_map[i]) for i in OCC_IDXS])
                u.err_ema=0.7*u.err_ema+0.3*err
                u.val_count += 1
                self.validated[p.h]+=1
                lr=0.2
                for i in OCC_IDXS:
                    j=idx_to_j[i]
                    u.pred_occ[j]=(1-lr)*u.pred_occ[j]+lr*obs_map[i]

        if not is_occluded:
            # clamp belief from observation
            obs_map={i:v for i,v in zip(obs_idx, obs_vals)}
            self.belief_occ=[float(obs_map[i]) for i in OCC_IDXS]
        self.t += 1

# -----------------------------
# Standard kNN scan baseline (cosine, replay)
# -----------------------------

class KNNScan:
    def __init__(self, k:int=7):
        self.k=k
        self.prev_row=[0]*H
        self.prev_col=[0]*W
        self.feat_len=(H+W)*2 + len(HORIZONS) + 3
        self.data={h: [] for h in HORIZONS}  # (feat, label_occ)
        self.pending: Dict[int, Tuple[int, List[float]]] = {}
        self.t=0

    def _shape_bits(self, obs_vals, obs_idx):
        left=mid=right=0
        for v,i in zip(obs_vals, obs_idx):
            if v==0: continue
            x=i%W
            if x < OCC_X0: left += 1
            elif x > OCC_X1: right += 1
            else: mid += 1
        return [1.0 if left>0 else 0.0, 1.0 if right>0 else 0.0, 1.0 if mid>0 else 0.0]

    def _features(self, obs_vals, obs_idx, h):
        row=[0]*H
        col=[0]*W
        for v,i in zip(obs_vals, obs_idx):
            if v==0: continue
            y=i//W; x=i%W
            row[y]+=1; col[x]+=1
        drow=[row[i]-self.prev_row[i] for i in range(H)]
        dcol=[col[i]-self.prev_col[i] for i in range(W)]
        self.prev_row=row; self.prev_col=col
        feats=[float(x) for x in (row+col+drow+dcol)]
        for hh in HORIZONS: feats.append(1.0 if hh==h else 0.0)
        feats += self._shape_bits(obs_vals, obs_idx)
        return feats

    def _cos(self,a,b):
        dot=na=nb=0.0
        for i in range(len(a)):
            dot += a[i]*b[i]
            na += a[i]*a[i]
            nb += b[i]*b[i]
        if na<=1e-12 or nb<=1e-12: return 0.0
        return dot/math.sqrt(na*nb)

    def predict(self, h, feat):
        buf=self.data[h]
        if not buf:
            return [0.0]*len(OCC_IDXS)
        sims=[]
        for f,y in buf:
            sims.append((self._cos(feat,f), y))
        sims.sort(key=lambda x:-x[0])
        top=sims[:self.k]
        out=[0.0]*len(OCC_IDXS)
        wsum=0.0
        for s,y in top:
            w=max(0.0,s)
            wsum += w
            for j in range(len(OCC_IDXS)):
                out[j]+=w*y[j]
        if wsum>1e-9:
            for j in range(len(OCC_IDXS)): out[j]/=wsum
        return out

    def on_occlusion_start(self, obs_vals, obs_idx, h_occ):
        feat=self._features(obs_vals, obs_idx, h_occ)
        self.pending[self.t+h_occ]=(h_occ, feat)
        return self.predict(h_occ, feat)

    def tick(self, obs_vals, obs_idx, is_reveal_tick: bool):
        if is_reveal_tick and self.t in self.pending:
            h, feat = self.pending.pop(self.t)
            obs_map={i:float(v) for i,v in zip(obs_idx, obs_vals)}
            label=[obs_map[i] for i in OCC_IDXS]
            self.data[h].append((feat, label))
        self.t += 1

# -----------------------------
# Run benchmark
# -----------------------------

def run(seed:int=0, T:int=5000):
    w=ChamberWorld(seed=seed, pre=6, post=2)

    nupca_b = LegacyBandAgent(full_scan=False, seed=seed+1)
    nupca_s = LegacyBandAgent(full_scan=True,  seed=seed+1)
    knn = KNNScan(k=7)

    iou_b={h:[] for h in HORIZONS}
    iou_s={h:[] for h in HORIZONS}
    iou_k={h:[] for h in HORIZONS}
    dist_b=[]; dist_s=[]; dist_k=[]

    t_b=t_s=t_k=0.0
    cur_h=None
    knn_pred=None

    for t in range(T):
        truth, occ, is_occ, is_start, is_reveal, h = w.step()
        obs_vals, obs_idx = w.observe(truth, occ)

        if is_start:
            cur_h=h
            nupca_b.on_occlusion_start(obs_vals, obs_idx, h)
            nupca_s.on_occlusion_start(obs_vals, obs_idx, h)
            knn_pred = knn.on_occlusion_start(obs_vals, obs_idx, h)

        # evaluate at reveal BEFORE learning on reveal tick
        if is_reveal and cur_h is not None and truth_in_occluder(truth):
            iou_b[cur_h].append(iou_occ(nupca_b.belief_occ, truth))
            iou_s[cur_h].append(iou_occ(nupca_s.belief_occ, truth))
            iou_k[cur_h].append(iou_occ(knn_pred if knn_pred is not None else [0.0]*len(OCC_IDXS), truth))

            ct = centroid(truth, OCC_IDXS)
            if ct is not None:
                pb = centroid_pred(nupca_b.belief_occ)
                ps = centroid_pred(nupca_s.belief_occ)
                pk = centroid_pred(knn_pred if knn_pred is not None else [0.0]*len(OCC_IDXS))
                if pb: dist_b.append(l2(pb, ct))
                if ps: dist_s.append(l2(ps, ct))
                if pk: dist_k.append(l2(pk, ct))

            cur_h=None
            knn_pred=None

        # tick agents
        t0=time.perf_counter(); nupca_b.tick(obs_vals, obs_idx, is_occ, is_reveal); t_b += time.perf_counter()-t0
        t0=time.perf_counter(); nupca_s.tick(obs_vals, obs_idx, is_occ, is_reveal); t_s += time.perf_counter()-t0
        t0=time.perf_counter(); knn.tick(obs_vals, obs_idx, is_reveal);           t_k += time.perf_counter()-t0

    print("\n================ CONCLUSIVE BENCHMARK ================")
    print(f"Steps={T}  horizons={HORIZONS}  grid={H}x{W}  occ=[{OCC_X0}..{OCC_X1}]")
    print("Metric: IoU on occluder region at reveal, conditioned on truth-in-occluder>0 (no empty IoU degeneracy).")
    print("Agents: NUPCA-Bounded (LSH), NUPCA-FullScan (oracle), kNN-Scan (standard replay baseline)\n")

    print("Speed:")
    print(f"  NUPCA-Bounded: {1000*t_b/T:.3f} ms/step")
    print(f"  NUPCA-FullScan:{1000*t_s/T:.3f} ms/step")
    print(f"  kNN-Scan:      {1000*t_k/T:.3f} ms/step\n")

    print("Retrieval health (NUPCA-Bounded):")
    hit_rate = nupca_b.stage1_hits / max(1, nupca_b.stage1_queries)
    print(f"  Stage1 hit-rate: {hit_rate:.3f}")
    print(f"  Avg candidates/query: {nupca_b.stage1_cand_sum / max(1,nupca_b.stage1_queries):.2f}")
    print(f"  Avg scored/query:     {nupca_b.stage2_scored_sum / max(1,nupca_b.stage1_queries):.2f}\n")

    print("Per-horizon IoU (mean±std, n):")
    print(f"{'h':>3s} {'Bounded':>10s} {'FullScan':>10s} {'kNN':>10s} {'n':>6s}")
    print("-"*44)
    for h in HORIZONS:
        n = len(iou_b[h])
        if n == 0:
            print(f"{h:3d} {'nan':>10s} {'nan':>10s} {'nan':>10s} {0:6d}")
            continue
        print(f"{h:3d} {mean(iou_b[h]):.3f}±{std(iou_b[h]):.3f} "
              f"{mean(iou_s[h]):.3f}±{std(iou_s[h]):.3f} "
              f"{mean(iou_k[h]):.3f}±{std(iou_k[h]):.3f} {n:6d}")

    print("\nCentroid distance during occluder-at-reveal (lower is better):")
    if dist_b: print(f"  Bounded:  {mean(dist_b):.2f} ± {std(dist_b):.2f} (n={len(dist_b)})")
    if dist_s: print(f"  FullScan: {mean(dist_s):.2f} ± {std(dist_s):.2f} (n={len(dist_s)})")
    if dist_k: print(f"  kNN:      {mean(dist_k):.2f} ± {std(dist_k):.2f} (n={len(dist_k)})")

if __name__ == "__main__":
    run(seed=0, T=5000)
