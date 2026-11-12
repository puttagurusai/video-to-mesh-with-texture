"""One textured surface mesh for the whole scene, from the fused keyframe depth.

  python -m video_to_3d.scene_mesh [--config pipeline_mouse.yaml] [--max-dim 256]

  1. TSDF fusion (GPU): every keyframe's depth is integrated into ONE signed-distance volume in the shared world.
     A voxel counts only if >= min_views keyframes observed it, so single-view noise never becomes surface.
  2. Marching cubes (scikit-image) -> a single mesh; tiny disconnected specks are dropped, a light Taubin smooth removes
     the voxel staircase.
  3. Multi-view texturing: each triangle takes its colour from the keyframe that sees it best (frontal view, high
     resolution, not occluded - checked against that keyframe's depth). Neighbouring triangles are nudged to share a
     photo to reduce seams. Triangles no keyframe sees stay grey.
  4. Outputs: <workdir>/geometry/scene_mesh.glb (textured, for Blender), scene_mesh.ply (vertex colours),
     <workdir>/viz/mesh_viewer.html (WebGL, works offline).

Honest limit: a TSDF can only be as consistent as the poses and depth that feed it. When keyframes disagree (the mouse
clip) the surface smears; the same code on a consistent capture (the hotel room) gives a clean room.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as TF
from skimage import measure

from .config import load_config
from .io_utils import Paths, read_json, record_metrics

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- volume
def load_depth(paths: Paths, k: int, conf_min: float, sil_thr: float = 0.0) -> np.ndarray:
    """Aligned depth with low-confidence pixels as NaN. With sil_thr > 0, pixels whose relative depth gradient per pixel
    exceeds it are also dropped: depth is blurred across object boundaries, and fusing those pixels stretches a skirt of
    false surface between foreground and background."""
    nm = paths.frame_name(k)
    Z = np.load(paths.depth__aligned / f"{nm}.npy")
    c = np.load(paths.depth__confidence / f"{nm}.npy").astype(np.float32)
    Z = np.where(np.isfinite(Z) & (c >= conf_min), Z, np.nan).astype(np.float32)
    if sil_thr > 0:
        Zf = np.nan_to_num(Z, nan=0.0)
        g = np.hypot(cv2.Sobel(Zf, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(Zf, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
        Z = np.where(g / np.maximum(Zf, 1e-6) > sil_thr, np.nan, Z)
    return Z


def scene_box(paths, frames, K, ids, conf_min, far_mult=3.0, stride=24):
    """Oriented (PCA-aligned) bounding box of the fused cloud, so a tilted room does not waste volume."""
    P = []
    for k in ids:
        Z = load_depth(paths, k, conf_min)
        h, w = Z.shape
        ys, xs = np.mgrid[stride // 2:h:stride, stride // 2:w:stride]
        z = Z[ys, xs]
        ok = np.isfinite(z) & (z < far_mult * np.nanmedian(Z))
        Xc = np.stack([(xs[ok] - K[0, 2]) / K[0, 0] * z[ok], (ys[ok] - K[1, 2]) / K[1, 1] * z[ok], z[ok]], 1)
        f = frames[k]
        P.append(Xc @ np.asarray(f["R_wc"]).T + np.asarray(f["t_wc"]))
    P = np.vstack(P)
    c0 = np.median(P, 0)
    _, _, Vt = np.linalg.svd(P - c0, full_matrices=False)
    b = (P - c0) @ Vt.T
    lo, hi = np.percentile(b, 0.5, axis=0), np.percentile(b, 99.5, axis=0)
    pad = 0.04 * (hi - lo)
    return c0, Vt, lo - pad, hi + pad, P


def fuse_tsdf(paths, frames, K, ids, c0, Vt, lo, dims, voxel, trunc, conf_min, sil_thr=0.0):
    nx, ny, nz = dims
    tsdf = torch.zeros(nx, ny, nz, device=DEV)
    wt = torch.zeros(nx, ny, nz, device=DEV)
    Vt_t = torch.as_tensor(Vt, dtype=torch.float32, device=DEV)
    c0_t = torch.as_tensor(c0, dtype=torch.float32, device=DEV)
    lo_t = torch.as_tensor(lo, dtype=torch.float32, device=DEV)
    iy = (torch.arange(ny, device=DEV, dtype=torch.float32) * voxel + lo_t[1])
    iz = (torch.arange(nz, device=DEV, dtype=torch.float32) * voxel + lo_t[2])
    gy, gz = torch.meshgrid(iy, iz, indexing="ij")
    cx_chunk = max(1, int(3_000_000 // (ny * nz)))
    fx, fy, cx, cy = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    for k in ids:
        Z = torch.from_numpy(load_depth(paths, k, conf_min, sil_thr)).to(DEV)
        h, w = Z.shape
        valid_img = torch.isfinite(Z).float()[None, None]
        Zimg = torch.nan_to_num(Z, nan=0.0)[None, None]
        f = frames[k]
        R = torch.as_tensor(np.asarray(f["R_cw"]), dtype=torch.float32, device=DEV)
        t = torch.as_tensor(np.asarray(f["t_cw"]), dtype=torch.float32, device=DEV)
        for x0 in range(0, nx, cx_chunk):
            x1 = min(nx, x0 + cx_chunk)
            gx = (torch.arange(x0, x1, device=DEV, dtype=torch.float32) * voxel + lo_t[0])[:, None, None].expand(-1, ny, nz)
            B = torch.stack([gx, gy[None].expand(x1 - x0, -1, -1), gz[None].expand(x1 - x0, -1, -1)], -1)
            X = c0_t + B @ Vt_t                                  # box coordinates -> world
            Xc = X @ R.T + t
            z = Xc[..., 2]
            u = fx * Xc[..., 0] / z + cx
            v = fy * Xc[..., 1] / z + cy
            inside = (z > 0.02) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
            grid = torch.stack([2 * u / (w - 1) - 1, 2 * v / (h - 1) - 1], -1).reshape(1, -1, 1, 2)
            zo = TF.grid_sample(Zimg, grid, mode="nearest", align_corners=True).reshape(z.shape)
            ov = TF.grid_sample(valid_img, grid, mode="nearest", align_corners=True).reshape(z.shape) > 0.5
            sdf = zo - z                                          # + in front of the surface (free space), - behind
            ok = inside & ov & (sdf > -trunc)
            val = torch.clamp(sdf / trunc, -1.0, 1.0)
            wn = ok.float()
            w_old = wt[x0:x1]
            tot = w_old + wn
            tsdf[x0:x1] = torch.where(tot > 0, (tsdf[x0:x1] * w_old + val * wn) / tot.clamp_min(1e-9), tsdf[x0:x1])
            wt[x0:x1] = tot
        del Z, Zimg, valid_img
    return tsdf.cpu().numpy(), wt.cpu().numpy()


# --------------------------------------------------------------------------- texturing
def best_frames(centroids, normals, frames, K, ids, paths, conf_min, tol_rel, tol_abs, min_cos=0.2):
    """Per face and keyframe: view-quality score (0 = cannot see it): frontal (cosine >= min_cos), close, unoccluded
    (depth test). Also returns, per face, the best cosine over keyframes whose image contains it (what any camera could
    ever see of it); a negative value means the face points away from every camera that looks at that spot."""
    F = len(centroids)
    scores = np.zeros((F, len(ids)), np.float16)
    cosmax = np.full(F, np.inf, np.float32)
    cen = torch.as_tensor(np.ascontiguousarray(centroids), dtype=torch.float32, device=DEV)
    nrm = torch.as_tensor(np.ascontiguousarray(normals), dtype=torch.float32, device=DEV)
    cos_best = torch.full((F,), -2.0, device=DEV)
    for j, k in enumerate(ids):
        f = frames[k]
        R = torch.as_tensor(np.asarray(f["R_cw"]), dtype=torch.float32, device=DEV)
        t = torch.as_tensor(np.asarray(f["t_cw"]), dtype=torch.float32, device=DEV)
        C = torch.as_tensor(np.asarray(f["t_wc"]), dtype=torch.float32, device=DEV)
        Z = torch.from_numpy(load_depth(paths, k, conf_min)).to(DEV)
        h, w = Z.shape
        Xc = cen @ R.T + t
        z = Xc[:, 2]
        u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
        v = K[1, 1] * Xc[:, 1] / z + K[1, 2]
        inside = (z > 0.02) & (u >= 2) & (u <= w - 3) & (v >= 2) & (v <= h - 3)
        to_cam = C - cen
        dist = to_cam.norm(dim=1).clamp_min(1e-9)
        cos = (nrm * to_cam).sum(1) / dist
        cos_best = torch.where(inside, torch.maximum(cos_best, cos), cos_best)
        ui = u.clamp(0, w - 1).round().long()
        vi = v.clamp(0, h - 1).round().long()
        zo = Z[vi, ui]
        seen = inside & (cos > min_cos) & torch.isfinite(zo) & ((z - zo).abs() <= tol_rel * z + tol_abs)
        sc = torch.where(seen, cos / (z * z), torch.zeros_like(z))
        scores[:, j] = sc.cpu().numpy().astype(np.float16)
    cb = cos_best.cpu().numpy()
    cosmax = np.where(cb > -1.5, cb, 1.0).astype(np.float32)       # never inside any image: keep (cannot judge)
    return scores, cosmax


def smooth_labels(scores, adj, iters=2, keep=0.5, topk=8):
    """Prefer one photo per patch: a face may switch to the photo most of its neighbours use, if that photo sees it well.
    Only each face's topk best keyframes are candidates, so memory is faces x topk, not faces x keyframes (a dense capture
    with 190 keyframes and a million faces would otherwise need gigabytes)."""
    F, nf = scores.shape
    k = min(topk, nf)
    s = scores.astype(np.float32)
    top = np.argpartition(-s, k - 1, axis=1)[:, :k]            # (F,k) candidate keyframes per face
    tv = np.take_along_axis(s, top, axis=1)                     # their scores
    best = tv.max(1)
    cand = (tv >= keep * best[:, None]) & (best[:, None] > 0)
    rows = np.arange(F)
    label = np.where(best > 0, top[rows, tv.argmax(1)], -1)
    a, b = adj[:, 0], adj[:, 1]
    slot = np.arange(k)[None, :]
    for _ in range(iters):
        counts = np.zeros(F * k, np.float32)
        for u, v in ((a, b), (b, a)):                           # each face counts how many neighbours use each candidate
            lv = label[v]
            match = (top[u] == lv[:, None]) & (lv[:, None] >= 0)
            counts += np.bincount((u[:, None] * k + slot)[match], minlength=F * k).astype(np.float32)
        pref = np.where(cand, counts.reshape(F, k) + 0.01 * tv / np.maximum(best[:, None], 1e-12), -1.0)
        label = np.where(best > 0, top[rows, pref.argmax(1)], -1)
    return label


# --------------------------------------------------------------------------- viewer
VIEWER = r"""<!doctype html><html><head><meta charset="utf-8"><title>Scene mesh</title><style>
html,body{margin:0;height:100%;background:#0f1115;color:#dde;font:13px system-ui,sans-serif;overflow:hidden}
canvas{display:block;width:100vw;height:100vh;cursor:grab}
#p{position:fixed;left:10px;top:10px;background:#1b1e26d9;padding:10px 12px;border-radius:8px;max-width:300px}
button,select{margin:3px 3px 0 0;background:#2a3040;color:#dde;border:1px solid #455;border-radius:4px;padding:3px 8px}
#s{margin-top:6px;color:#9ab}</style></head><body><canvas id="c"></canvas>
<div id="p"><b>__TITLE__</b><div><label>view: <select id="mode"><option value="0">textured</option><option value="1">shaded (geometry only)</option>
<option value="2">colour by source keyframe</option></select></label></div>
<div><button id="b0">camera view</button><button id="b1">oblique</button><button id="b2">top</button><button id="b3">side</button></div>
<div style="margin-top:4px">look from keyframe camera: <input type="range" id="kf" min="0" max="0" value="0" style="width:150px"> <span id="kfl"></span></div>
<div id="s"></div><div style="margin-top:4px;color:#789">drag = orbit, shift/right-drag = pan, wheel = zoom</div></div>
<script>
const D=__DATA__;
const cv=document.getElementById('c'),gl=cv.getContext('webgl2',{antialias:true});
if(!gl){document.getElementById('s').textContent='WebGL2 is not available in this browser.';}
const vsrc=`#version 300 es
in vec3 aPos;in vec2 aUV;uniform vec3 uEye,uR,uU,uF;uniform float uTan,uAsp,uNear,uFar;out vec2 vUV;out vec3 vPos;
void main(){vUV=aUV;vPos=aPos;vec3 v=aPos-uEye;float x=dot(v,uR),y=dot(v,uU),z=dot(v,uF);
gl_Position=vec4(x/(uTan*uAsp),y/uTan,(z*(uFar+uNear)-2.0*uFar*uNear)/(uFar-uNear),z);}`;
const fsrc=`#version 300 es
precision highp float;in vec2 vUV;in vec3 vPos;uniform sampler2D uTex;uniform int uMode,uHas;uniform vec3 uTint;out vec4 o;
void main(){if(uMode==0){o=uHas==1?vec4(texture(uTex,vUV).rgb,1.0):vec4(0.5,0.5,0.5,1.0);}
else if(uMode==1){vec3 n=normalize(cross(dFdx(vPos),dFdy(vPos)));float l=0.3+0.7*abs(dot(n,normalize(vec3(0.3,0.8,0.5))));o=vec4(vec3(0.82)*l,1.0);}
else{o=vec4(uTint,1.0);}}`;
function sh(t,s){const x=gl.createShader(t);gl.shaderSource(x,s);gl.compileShader(x);if(!gl.getShaderParameter(x,gl.COMPILE_STATUS))console.error(gl.getShaderInfoLog(x));return x}
const prog=gl.createProgram();gl.attachShader(prog,sh(gl.VERTEX_SHADER,vsrc));gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,fsrc));
gl.bindAttribLocation(prog,0,'aPos');gl.bindAttribLocation(prog,1,'aUV');gl.linkProgram(prog);gl.useProgram(prog);
const U=n=>gl.getUniformLocation(prog,n);
function buf(t,data){const b=gl.createBuffer();gl.bindBuffer(t,b);gl.bufferData(t,data,gl.STATIC_DRAW);return b}
const b64=s=>{const bin=atob(s),n=bin.length,u=new Uint8Array(n);for(let i=0;i<n;i++)u[i]=bin.charCodeAt(i);return u.buffer};
const chunks=D.chunks.map((c,i)=>{const vao=gl.createVertexArray();gl.bindVertexArray(vao);
 buf(gl.ARRAY_BUFFER,new Float32Array(b64(c.pos)));gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
 buf(gl.ARRAY_BUFFER,new Float32Array(b64(c.uv)));gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,2,gl.FLOAT,false,0,0);
 const ib=buf(gl.ELEMENT_ARRAY_BUFFER,new Uint32Array(b64(c.idx)));gl.bindVertexArray(null);
 const ch={vao,n:c.n,tex:null,has:0,tint:[...Array(3)].map((_,k)=>0.25+0.7*Math.abs(Math.sin(i*2.399+k*2.1)))};
 if(c.tex){const img=new Image();img.onload=()=>{const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,img);gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR_MIPMAP_LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);ch.tex=t;ch.has=1;draw()};img.src='data:image/jpeg;base64,'+c.tex}
 return ch});
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]],dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2],
 nrm=a=>{const l=Math.hypot(...a)||1;return[a[0]/l,a[1]/l,a[2]/l]},crs=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const UP=nrm(D.up);let e1=nrm(crs(UP,Math.abs(UP[0])<0.9?[1,0,0]:[0,1,0]));const e2=crs(UP,e1);e1=crs(e2,UP);
let T,dist,th,ph,fov=D.fov;
function setView(eye,target,f){T=target.slice();const v=sub(eye,T);dist=Math.hypot(...v);ph=Math.asin(Math.max(-1,Math.min(1,dot(v,UP)/dist)));
 th=Math.atan2(dot(v,e2),dot(v,e1));fov=f}
const camView=()=>setView(D.cam.eye,D.cam.target,D.fov);
camView();
function eyePos(){const c=Math.cos(ph);return[0,1,2].map(i=>T[i]+dist*(c*(Math.cos(th)*e1[i]+Math.sin(th)*e2[i])+Math.sin(ph)*UP[i]))}
function basis(){const eye=eyePos(),f=nrm(sub(T,eye)),r=nrm(crs(f,UP)),u=crs(r,f);return{eye,f,r,u}}
function draw(){if(!gl)return;const dpr=devicePixelRatio;const W=cv.width=innerWidth*dpr,H=cv.height=innerHeight*dpr;gl.viewport(0,0,W,H);
 gl.clearColor(0.06,0.065,0.08,1);gl.enable(gl.DEPTH_TEST);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
 const B=basis();gl.uniform3fv(U('uEye'),B.eye);gl.uniform3fv(U('uR'),B.r);gl.uniform3fv(U('uU'),B.u);gl.uniform3fv(U('uF'),B.f);
 gl.uniform1f(U('uTan'),Math.tan(fov/2));gl.uniform1f(U('uAsp'),W/H);gl.uniform1f(U('uNear'),D.scale*0.02);gl.uniform1f(U('uFar'),D.scale*60);
 const mode=+document.getElementById('mode').value;gl.uniform1i(U('uMode'),mode);gl.uniform1i(U('uTex'),0);
 for(const ch of chunks){gl.uniform1i(U('uHas'),ch.has);gl.uniform3fv(U('uTint'),ch.tint);if(ch.tex){gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,ch.tex)}
  gl.bindVertexArray(ch.vao);gl.drawElements(gl.TRIANGLES,ch.n*3,gl.UNSIGNED_INT,0)}}
let drag=null;cv.onmousedown=e=>{drag={x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===2};cv.style.cursor='grabbing'};
addEventListener('mouseup',()=>{drag=null;cv.style.cursor='grab'});cv.oncontextmenu=e=>e.preventDefault();
addEventListener('mousemove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
 if(drag.pan){const B=basis(),k=dist*0.0016;for(let i=0;i<3;i++)T[i]+=(-dx*B.r[i]+dy*B.u[i])*k}
 else{th-=dx*0.005;ph=Math.max(-1.5,Math.min(1.5,ph+dy*0.005))}draw()});
cv.onwheel=e=>{e.preventDefault();dist*=Math.exp(e.deltaY*0.001);draw()};addEventListener('resize',draw);
document.getElementById('mode').onchange=draw;
const ctr=D.center,s=D.scale,r0=D.cam.right,f0=D.cam.fwd;
document.getElementById('b0').onclick=()=>{camView();draw()};
document.getElementById('b1').onclick=()=>{setView([ctr[0]+0.9*s*r0[0]+0.9*s*UP[0]-1.8*s*f0[0],ctr[1]+0.9*s*r0[1]+0.9*s*UP[1]-1.8*s*f0[1],ctr[2]+0.9*s*r0[2]+0.9*s*UP[2]-1.8*s*f0[2]],ctr,1.0);draw()};
document.getElementById('b2').onclick=()=>{setView([ctr[0]+3.2*s*UP[0],ctr[1]+3.2*s*UP[1],ctr[2]+3.2*s*UP[2]],ctr,1.0);th+=0.0;draw()};
document.getElementById('b3').onclick=()=>{setView([ctr[0]+3.2*s*r0[0],ctr[1]+3.2*s*r0[1],ctr[2]+3.2*s*r0[2]],ctr,1.0);draw()};
const kf=document.getElementById('kf');kf.max=D.cams.length-1;
kf.oninput=()=>{const c=D.cams[+kf.value];setView(c.eye,c.target,D.fov);document.getElementById('kfl').textContent='keyframe '+c.frame;draw()};
document.getElementById('s').textContent=D.note;draw();
</script></body></html>"""


# --------------------------------------------------------------------------- main build
def build(cfg, paths: Paths, max_dim: int | None = None, min_views: int | None = None, tex_side: int | None = None,
          label: str = "") -> dict:
    import trimesh
    from PIL import Image
    t_start = time.time()
    timing = {}
    mc = cfg.mesh
    max_dim, min_views, tex_side = max_dim or mc.max_dim, min_views or mc.min_views, tex_side or mc.tex_side
    sc = cfg.surface
    poses = read_json(paths.tracking / "camera_poses.json")
    frames = {f["frame"]: f for f in poses["frames"]}
    cam = next(iter(poses["cameras"].values()))
    K = np.asarray(cam["K"])
    ids_all = sorted(frames)
    conf_min = sc.conf_min

    # keyframes whose depth disagrees with their neighbours would smear the fused surface: leave them out
    ids, excluded = list(ids_all), []
    val_path = paths.viz / "validation.json"
    if val_path.exists():
        per_frame = read_json(val_path).get("stage3_depth", {}).get("per_frame", [])
        bad = {e["frame"] for e in per_frame if e.get("frac_consistent_within_tol", 1.0) < mc.min_frame_consistency}
        if bad and len(ids_all) - len(bad) >= 6:
            ids = [k for k in ids_all if k not in bad]
            excluded = sorted(bad)

    t0 = time.time()
    c0, Vt, lo, hi, cloud = scene_box(paths, frames, K, ids, conf_min)
    ext = hi - lo
    zmed = float(np.nanmedian(load_depth(paths, ids[0], conf_min)))
    for dim in dict.fromkeys([max_dim, int(max_dim * 0.75), int(max_dim * 0.5)]):   # GPU out of memory -> coarser volume
        voxel = float(ext.max() / dim)
        dims = tuple(int(np.ceil(e / voxel)) + 1 for e in ext)
        trunc = max(3.0 * voxel, 0.05 * zmed)
        try:
            tsdf, wt = fuse_tsdf(paths, frames, K, ids, c0, Vt, lo, dims, voxel, trunc, conf_min, mc.silhouette_thr)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[mesh] GPU out of memory at {dim} voxels, retrying coarser")
    else:
        raise RuntimeError("GPU out of memory even at half the requested volume resolution; lower mesh.max_dim")
    timing["tsdf_fusion_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    from scipy.ndimage import gaussian_filter
    mask = wt >= min_views
    # smooth the VOLUME (not the mesh): Taubin smoothing on the mesh stretched vertices at hole boundaries into huge
    # spikes (edges of 20 voxels), whereas marching cubes on a smoothed volume cannot create them
    obs = mask.astype(np.float32)
    num, den = gaussian_filter(tsdf * obs, 1.0), gaussian_filter(obs, 1.0)
    tsdf_s = np.where(den > 1e-3, num / np.maximum(den, 1e-6), tsdf)
    verts, faces, _, _ = measure.marching_cubes(tsdf_s, level=0.0, spacing=(voxel,) * 3, mask=mask, allow_degenerate=False)
    Xw = c0 + (verts + lo) @ Vt
    mesh = trimesh.Trimesh(vertices=Xw, faces=faces, process=False)
    n_raw = len(mesh.faces)
    comps = mesh.split(only_watertight=False)
    keep = [c for c in comps if len(c.faces) >= max(200, 0.003 * n_raw)]
    mesh = trimesh.util.concatenate(keep) if keep else mesh
    tri = mesh.vertices[mesh.faces]
    longest = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1)
    if (longest > 3 * voxel).any():                    # safety net: nothing marching cubes makes is this long
        mesh.update_faces(longest <= 3 * voxel)
        mesh.remove_unreferenced_vertices()
    # make the winding point to the free-space side. The volume itself says which side that is (tsdf > 0 = in front of
    # the surface), which is exact for any scene. The earlier "nearest camera" guess failed on an orbit capture: the whole
    # mesh came out inside-out and the unseen-face filter then deleted 88% of it as back-facing.
    from scipy.ndimage import map_coordinates
    cen = mesh.triangles_center
    pick = np.arange(0, len(cen), max(1, len(cen) // 20000))
    nrm_s = mesh.face_normals[pick]
    ahead = ((cen[pick] + 0.7 * voxel * nrm_s - c0) @ Vt.T - lo) / voxel
    behind = ((cen[pick] - 0.7 * voxel * nrm_s - c0) @ Vt.T - lo) / voxel
    t_ahead = map_coordinates(tsdf_s, ahead.T, order=1, mode="nearest")
    t_behind = map_coordinates(tsdf_s, behind.T, order=1, mode="nearest")
    free_frac = float(np.mean(t_ahead > t_behind))
    print(f"[mesh] {free_frac:.0%} of faces point to the free side as produced; {'inverting' if free_frac < 0.5 else 'keeping'} orientation")
    if free_frac < 0.5:
        # NOT mesh.invert(): in trimesh 4.x it flips the faces but face_normals keeps returning the old (stale) cached
        # normals, which made the visibility test below delete ~90% of a correct mesh as "back-facing". Rebuild instead.
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=np.fliplr(mesh.faces), process=False)
    timing["marching_cubes_clean_s"] = round(time.time() - t0, 1)

    # ---- texturing
    t0 = time.time()
    cen, nrm = mesh.triangles_center, mesh.face_normals
    n_faces_0 = len(cen)
    scores, cosmax = best_frames(cen, nrm, frames, K, ids, paths, conf_min, 0.04, 2.0 * voxel, min_cos=mc.tex_min_cos)
    first_pass = float((scores.max(1) > 0).mean())
    unt = scores.max(1) == 0
    if unt.any():                                          # grazing faces nobody saw frontally: relaxed angle, same depth test
        s2, _ = best_frames(cen[unt], nrm[unt], frames, K, ids, paths, conf_min, 0.04, 2.0 * voxel, min_cos=mc.tex_fallback_cos)
        scores[unt] = s2
    dropped = 0
    if mc.drop_unseen_faces:                               # faces pointing away from every camera that looks at them
        drop = (scores.max(1) == 0) & (cosmax < 0.0)
        dropped = int(drop.sum())
        if dropped:
            mesh.update_faces(~drop)
            mesh.remove_unreferenced_vertices()
            scores = scores[~drop]
            cosmax = cosmax[~drop]
    grazing = float((cosmax < 0.2).mean())                 # faces no keyframe ever saw head-on: stretched skirts, steep walls
    label_f = smooth_labels(scores, mesh.face_adjacency, iters=2)
    textured = float((label_f >= 0).mean())
    timing["texturing_s"] = round(time.time() - t0, 1)

    # ---- chunks (one per source keyframe), texture = that keyframe's photo
    t0 = time.time()
    chunks, glb_scene = [], trimesh.Scene()
    F = mesh.faces
    Vw = np.asarray(mesh.vertices)
    vertex_rgb = np.full((len(Vw), 3), 128, np.uint8)
    for j in list(range(len(ids))) + [-1]:
        fsel = np.flatnonzero(label_f == j)
        if len(fsel) == 0:
            continue
        used, inv = np.unique(F[fsel], return_inverse=True)
        pos = Vw[used].astype(np.float32)
        idx = inv.reshape(-1, 3).astype(np.uint32)
        if j >= 0:
            k = ids[j]
            f = frames[k]
            Xc = pos @ np.asarray(f["R_cw"]).T.astype(np.float32) + np.asarray(f["t_cw"], np.float32)
            u = (K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2]) / cam["width"]
            v = (K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]) / cam["height"]
            uv = np.clip(np.stack([u, v], 1), 0, 1).astype(np.float32)
            img = cv2.imread(str(paths.keyframes / f"{paths.frame_name(k)}.png"))
            vertex_rgb[used] = img[np.clip((v * img.shape[0]).astype(int), 0, img.shape[0] - 1),
                                   np.clip((u * img.shape[1]).astype(int), 0, img.shape[1] - 1)][:, ::-1]
            s = tex_side / max(img.shape[:2])
            small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
            jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()
            tex_b64 = base64.b64encode(jpg).decode()
            pil = Image.open(io.BytesIO(jpg))
        else:
            uv, tex_b64, pil = np.zeros((len(pos), 2), np.float32), None, None
        chunks.append({"pos": base64.b64encode(pos.tobytes()).decode(), "uv": base64.b64encode(uv.tobytes()).decode(),
                       "idx": base64.b64encode(idx.tobytes()).decode(), "tex": tex_b64, "n": int(len(idx))})
        gm = trimesh.Trimesh(vertices=pos * np.array([1, -1, -1], np.float32), faces=idx, process=False)   # CV -> glTF axes
        if pil is not None:
            gm.visual = trimesh.visual.TextureVisuals(uv=np.stack([uv[:, 0], 1 - uv[:, 1]], 1), image=pil)
        glb_scene.add_geometry(gm, node_name=f"keyframe_{ids[j] if j >= 0 else 'untextured'}")
    (paths.root / "geometry").mkdir(exist_ok=True)
    glb_ok = True
    try:
        glb_scene.export(paths.root / "geometry" / "scene_mesh.glb")
    except Exception as exc:
        glb_ok = False
        print(f"[mesh] GLB export failed: {type(exc).__name__}: {exc}")
    trimesh.Trimesh(vertices=Vw * np.array([1, -1, -1]), faces=F, vertex_colors=np.hstack([vertex_rgb, np.full((len(Vw), 1), 255, np.uint8)]),
                    process=False).export(paths.root / "geometry" / "scene_mesh.ply")

    # ---- viewer
    ups = np.mean([np.asarray(frames[k]["R_wc"]) @ np.array([0.0, -1.0, 0.0]) for k in ids], axis=0)
    f0 = frames[ids[0]]
    Rwc0, C0 = np.asarray(f0["R_wc"]), np.asarray(f0["t_wc"])
    scale = float(zmed)
    center = np.median(Vw, 0)
    note = (f"{len(F):,} triangles, {len(Vw):,} vertices, {textured:.0%} textured from {len(ids)} keyframes; "
            f"voxel {voxel:.4f} (= {100 * voxel / zmed:.1f}% of median depth)")
    data = {"chunks": chunks, "up": ups.tolist(), "fov": float(2 * np.arctan(cam["height"] / 2 / K[1, 1])), "scale": scale,
            "center": center.tolist(), "note": note,
            "cams": [{"frame": int(k), "eye": np.asarray(frames[k]["t_wc"]).tolist(),
                      "target": (np.asarray(frames[k]["t_wc"]) + np.asarray(frames[k]["R_wc"])[:, 2] * zmed).tolist()} for k in ids],
            "cam": {"eye": C0.tolist(), "target": (C0 + Rwc0[:, 2] * zmed).tolist(), "right": Rwc0[:, 0].tolist(),
                    "fwd": Rwc0[:, 2].tolist()}}
    html = VIEWER.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__TITLE__", f"Scene mesh {label}")
    (paths.viz / "mesh_viewer.html").write_text(html, encoding="utf-8")
    timing["export_viewer_s"] = round(time.time() - t0, 1)

    # ---- agreement between mesh and the fused cloud
    from scipy.spatial import cKDTree
    tm = cKDTree(Vw)
    sub = cloud[:: max(1, len(cloud) // 150000)]
    d_pts, _ = tm.query(sub)
    tc = cKDTree(sub)
    d_mesh, _ = tc.query(Vw[:: max(1, len(Vw) // 150000)])
    stats = {"voxel": voxel, "voxel_over_median_depth": voxel / zmed, "grid": list(dims), "min_views": min_views,
             "triangles": int(len(F)), "vertices": int(len(Vw)), "triangles_before_cleanup": int(n_raw),
             "textured_fraction": textured, "textured_first_pass": first_pass, "grazing_fraction": grazing,
             "faces_dropped_pointing_away_from_all_cameras": dropped, "faces_before_drop": n_faces_0,
             "keyframes_used": len(ids), "keyframes_excluded_for_inconsistency": excluded,
             "texture_chunks": int(sum(1 for c in chunks if c["tex"])),
             "cloud_points_within_2_voxels_of_mesh": float((d_pts <= 2 * voxel).mean()),
             "mesh_vertices_within_2_voxels_of_cloud": float((d_mesh <= 2 * voxel).mean()),
             "glb_written": glb_ok, "timing_s": {**timing, "total_s": round(time.time() - t_start, 1)}}
    record_metrics(paths, "scene_mesh", stats)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline.yaml")
    ap.add_argument("--max-dim", type=int, default=None, help="override mesh.max_dim")
    ap.add_argument("--min-views", type=int, default=None, help="override mesh.min_views")
    a = ap.parse_args()
    cfg = load_config(a.config)
    paths = Paths(cfg)
    st = build(cfg, paths, max_dim=a.max_dim, min_views=a.min_views, label=f"({paths.root.name})")
    print(f"mesh: {st['triangles']:,} triangles / {st['vertices']:,} vertices (voxel {100 * st['voxel_over_median_depth']:.1f}% of depth, "
          f"grid {st['grid']}), {st['textured_fraction']:.0%} textured from {st['texture_chunks']} keyframes "
          f"(first pass {st['textured_first_pass']:.0%}; {st['faces_dropped_pointing_away_from_all_cameras']:,} faces dropped as unseen; "
          f"{len(st['keyframes_excluded_for_inconsistency'])} keyframes excluded)")
    print(f"agreement with fused cloud: {st['cloud_points_within_2_voxels_of_mesh']:.0%} of cloud points within 2 voxels of the mesh, "
          f"{st['mesh_vertices_within_2_voxels_of_cloud']:.0%} of mesh vertices within 2 voxels of the cloud")
    print("time:", st["timing_s"])
