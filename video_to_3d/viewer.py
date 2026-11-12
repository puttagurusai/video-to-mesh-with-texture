"""Standalone 3D viewer for one frame's provisional edges:  python -m video_to_3d.viewer --frame 13

Writes viz/viewer_frameNNNN.html (no external libraries, works offline). Shows the stored 3D edges coloured by shape,
an optional coloured depth point cloud of the same frame for context, the camera path, and the source image.
Drag = orbit, right-drag / shift-drag = pan, wheel = zoom. World = DA3 first-camera frame (arbitrary units).
"""
from __future__ import annotations

import argparse
import base64
import json

import cv2
import numpy as np

from .config import load_config
from .io_utils import Paths, read_json

CONFIG = "pipeline.yaml"   # set from --config

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>3D edges frame __FRAME__</title>
<style>
html,body{margin:0;height:100%;background:#101216;color:#dde;font:13px system-ui,sans-serif;overflow:hidden}
canvas{display:block;width:100vw;height:100vh;cursor:grab}
#panel{position:fixed;left:10px;top:10px;background:#1b1e26cc;padding:10px 12px;border-radius:8px;max-width:260px}
#panel h3{margin:0 0 6px;font-size:13px} label{display:block;margin:2px 0;cursor:pointer}
.sw{display:inline-block;width:12px;height:12px;margin-right:6px;vertical-align:-2px;border-radius:2px}
button{margin:4px 4px 0 0;background:#2a3040;color:#dde;border:1px solid #455;border-radius:4px;padding:3px 8px;cursor:pointer}
img{display:block;margin-top:8px;width:100%;border-radius:4px} #stats{margin-top:6px;color:#9ab}
</style></head><body><canvas id="c"></canvas>
<div id="panel"><h3>Frame __FRAME__ &middot; provisional 3D edges</h3><div id="shapes"></div><hr style="border-color:#334">
<div id="kinds"></div><hr style="border-color:#334">
<label id="nodeRow" style="display:none"><input type="checkbox" id="nodes" checked> graph vertices (red = junction)</label>
<label id="loopRow" style="display:none"><input type="checkbox" id="loops" checked> loops (green = depth-supported)</label>
<label id="surfRow" style="display:none"><input type="checkbox" id="surf" checked> plane surfaces (depth-supported)</label>
<label><input type="checkbox" id="pts" checked> depth points</label>
<label>&nbsp;&nbsp;colour: <select id="pmode"><option value="rgb">image colour</option><option value="frame">source keyframe</option><option value="agree">multi-view agreement</option></select></label>
<label><input type="checkbox" id="cams" checked> camera path</label>
<button id="vcam">camera view</button><button id="vobl">oblique</button><button id="vtop">top</button><button id="vside">side</button>
<div id="stats"></div><div id="note" style="margin-top:4px;color:#9ab"></div><img src="data:image/jpeg;base64,__IMG__"></div>
<script>
const D=__DATA__;
const SHAPES={straight:'#2f81ff',curved:'#ff8c00',piecewise:'#18d018',undetermined:'#8a8a8a'};
const showShape={straight:true,curved:true,piecewise:true,undetermined:true};
const showKind={};D.kinds.forEach(k=>showKind[k]=true);
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
let W=0,H=0;function resize(){W=cv.width=innerWidth*devicePixelRatio;H=cv.height=innerHeight*devicePixelRatio}addEventListener('resize',()=>{resize();draw()});resize();
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]],dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const norm=a=>{const l=Math.hypot(...a)||1;return[a[0]/l,a[1]/l,a[2]/l]};
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
let T,dist,yaw,pitch;
function setView(eye,target){T=target.slice();const v=sub(eye,T);dist=Math.hypot(...v);yaw=Math.atan2(v[0],v[2]);pitch=Math.asin(v[1]/dist)}
let fov=D.fov;
function camView(){setView(D.cam.eye,D.cam.target);fov=D.fov}
function orbitView(eye,target){setView(eye,target);fov=1.1}
camView();
function basis(){const eye=[T[0]+dist*Math.cos(pitch)*Math.sin(yaw),T[1]+dist*Math.sin(pitch),T[2]+dist*Math.cos(pitch)*Math.cos(yaw)];
 const f=norm(sub(T,eye)),r=norm(cross([0,1,0],f)),u=cross(f,r);return{eye,f,r,u}}
function draw(){const B=basis(),fl=H*0.5/Math.tan(0.5*fov),cx=W/2,cy=H/2,near=0.02*D.scale;
 ctx.fillStyle='#101216';ctx.fillRect(0,0,W,H);
 const P=p=>{const v=sub(p,B.eye);return[dot(v,B.r),dot(v,B.u),dot(v,B.f)]};
 const S=c=>[cx+fl*c[0]/c[2],cy-fl*c[1]/c[2]];
 if(document.getElementById('pts').checked){const a=D.pts,pm=document.getElementById('pmode').value;for(let i=0;i<a.length;i+=8){const c=P([a[i],a[i+1],a[i+2]]);if(c[2]<near)continue;
   const s=S(c);let col;
   if(pm==='frame')col='hsla('+Math.round(300*a[i+6]/Math.max(1,D.nframes-1))+',85%,55%,0.6)';
   else if(pm==='agree'){const v=a[i+7];col=v<0?'rgba(130,130,130,0.5)':'rgba('+Math.round(255*(1-v/100))+','+Math.round(220*v/100)+',30,0.65)'}
   else col='rgba('+a[i+3]+','+a[i+4]+','+a[i+5]+',0.45)';
   ctx.fillStyle=col;const z=Math.max(1,fl*0.0022*D.scale/c[2]);ctx.fillRect(s[0],s[1],z,z)}}
 const seg=(p,q)=>{let a=P(p),b=P(q);if(a[2]<near&&b[2]<near)return;
   if(a[2]<near){const t=(near-a[2])/(b[2]-a[2]);a=[a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]),near]}
   else if(b[2]<near){const t=(near-b[2])/(a[2]-b[2]);b=[b[0]+t*(a[0]-b[0]),b[1]+t*(a[1]-b[1]),near]}
   const s=S(a),e=S(b);ctx.moveTo(s[0],s[1]);ctx.lineTo(e[0],e[1])};
 if(D.quads.length&&document.getElementById('surf').checked){const qs=[];
   for(const q of D.quads){const pr=q.c.map(P);if(pr.some(c=>c[2]<near))continue;qs.push([pr.reduce((a,c)=>a+c[2],0)/4,pr.map(S),q.col])}
   qs.sort((a,b)=>b[0]-a[0]);for(const [d,s,col] of qs){ctx.beginPath();ctx.moveTo(s[0][0],s[0][1]);for(let i=1;i<4;i++)ctx.lineTo(s[i][0],s[i][1]);ctx.closePath();
     ctx.fillStyle='rgba('+col[0]+','+col[1]+','+col[2]+',0.55)';ctx.fill()}}
 if(document.getElementById('cams').checked){ctx.strokeStyle='#ffd24d';ctx.lineWidth=1.5*devicePixelRatio;ctx.beginPath();
   for(let i=1;i<D.path.length;i++)seg(D.path[i-1],D.path[i]);
   for(const q of D.cam.frustum){seg(D.cam.eye,q)}for(let i=0;i<4;i++)seg(D.cam.frustum[i],D.cam.frustum[(i+1)%4]);ctx.stroke()}
 let n=0;for(const e of D.edges){if(!showShape[e.t]||!showKind[e.k])continue;n++;
   ctx.strokeStyle=SHAPES[e.t];ctx.globalAlpha=0.45+0.55*Math.min(1,e.c/0.5);ctx.lineWidth=(e.t==='undetermined'?1.2:2.4)*devicePixelRatio;ctx.beginPath();
   for(let i=1;i<e.v.length;i++)seg(e.v[i-1],e.v[i]);ctx.stroke()}
 ctx.globalAlpha=1;
 if(D.loops.length&&document.getElementById('loops').checked){const LC={supported:'0,210,0',opening:'255,60,60',mismatch:'255,160,0',unobserved:'120,170,255'};
   for(const l of D.loops){const pr=l.ring.map(P);if(pr.some(c=>c[2]<near))continue;const s=pr.map(S);
     ctx.beginPath();s.forEach((q,i)=>i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));ctx.closePath();
     ctx.fillStyle='rgba('+LC[l.cls]+',0.28)';ctx.fill();ctx.strokeStyle='rgb('+LC[l.cls]+')';ctx.lineWidth=2.2*devicePixelRatio;ctx.stroke()}}
 if(D.nodes.length&&document.getElementById('nodes').checked){for(const nd of D.nodes){const c=P(nd);if(c[2]<near)continue;const s=S(c);
   ctx.fillStyle=nd[3]>=3?'#ff3030':nd[3]===2?'#ffffff':'#8a8a8a';const r=(nd[3]>=3?4.5:3)*devicePixelRatio;ctx.beginPath();ctx.arc(s[0],s[1],r,0,6.283);ctx.fill()}}
 document.getElementById('stats').textContent=n+' of '+D.edges.length+' edges shown'+(D.nodes.length?(' · '+D.nodes.length+' vertices · '+D.loops.length+' loops'):'')}
let drag=null;cv.onmousedown=e=>{drag={x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===2};cv.style.cursor='grabbing'};
addEventListener('mouseup',()=>{drag=null;cv.style.cursor='grab'});cv.oncontextmenu=e=>e.preventDefault();
addEventListener('mousemove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
 if(drag.pan){const B=basis(),k=dist*0.0016;for(let i=0;i<3;i++)T[i]+=(-dx*B.r[i]+dy*B.u[i])*k}
 else{yaw-=dx*0.005;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*0.005))}draw()});
cv.onwheel=e=>{e.preventDefault();dist*=Math.exp(e.deltaY*0.001);draw()};
const mk=(id,obj,colors)=>{const el=document.getElementById(id);for(const k in obj){const l=document.createElement('label');const cnt=D.counts[id][k]||0;
 l.innerHTML='<input type="checkbox" checked> '+(colors?'<span class="sw" style="background:'+colors[k]+'"></span>':'')+k+' <span style="color:#789">('+cnt+')</span>';
 l.firstChild.onchange=e=>{obj[k]=e.target.checked;draw()};el.appendChild(l)}};
mk('shapes',showShape,SHAPES);mk('kinds',showKind,null);
for(const id of['pts','cams','nodes','loops','surf','pmode'])document.getElementById(id).onchange=draw;
document.getElementById('note').textContent=D.note||'';
if(D.quads.length)document.getElementById('surfRow').style.display='block';
if(D.nodes.length)document.getElementById('nodeRow').style.display='block';
if(D.loops.length)document.getElementById('loopRow').style.display='block';
document.getElementById('vcam').onclick=()=>{camView();draw()};
document.getElementById('vtop').onclick=()=>{orbitView([D.center[0],D.center[1]+D.scale*3.0,D.center[2]-0.001],D.center);draw()};
document.getElementById('vside').onclick=()=>{orbitView([D.center[0]+D.scale*3.0,D.center[1],D.center[2]],D.center);draw()};
document.getElementById('vobl').onclick=()=>{const c=D.cam.eye;orbitView([c[0]+D.scale*0.9,c[1]+D.scale*0.7,c[2]-D.scale*1.3],D.center);draw()};
draw();
</script></body></html>"""


def build_map() -> str:
    """Viewer for the persistent map (stage 8): all map edges, coloured by shape, filterable by status."""
    cfg = load_config(CONFIG)
    paths = Paths(cfg)
    poses = read_json(paths.tracking / "camera_poses.json")
    fr = {f["frame"]: f for f in poses["frames"]}
    cam = next(iter(poses["cameras"].values()))
    K = np.array(cam["K"])
    disp = lambda P: np.asarray(P) * np.array([1.0, -1.0, 1.0])  # noqa: E731
    map_path = paths.tracking / "edge_map.json"
    M = read_json(map_path)["edges"] if map_path.exists() else []     # a run that stopped before stage 8 shows its cloud only
    E = [{"t": e["type"], "k": e["status"] + (" (3+ views)" if len(e["observed_frames"]) >= 3 else ""),
          "c": e["confidence"], "v": np.round(disp(e["vertices_3d"]), 4).tolist()} for e in M]
    kinds = sorted({e["k"] for e in E})
    counts = {"shapes": {}, "kinds": {}}
    for e in E:
        counts["shapes"][e["t"]] = counts["shapes"].get(e["t"], 0) + 1
        counts["kinds"][e["k"]] = counts["kinds"].get(e["k"], 0) + 1
    # the COMPLETE scene cloud: every keyframe's depth lifted into the shared world (random subset for the browser),
    # each point carrying its source keyframe and how many other keyframes confirm it
    from .scene_cloud import build as build_cloud
    cloud = build_cloud(cfg, paths)
    sub = cloud["sub"]
    agree = np.where(np.isfinite(cloud["support"]), np.round(100 * cloud["support"]), -1)
    allw = cloud["P"][sub]
    pts = [np.concatenate([np.round(disp(allw), 4), cloud["col"][sub], cloud["src"][sub][:, None], agree[:, None]], 1).ravel()]
    cs = cloud["stats"]
    note = (f"cloud: all {cs['keyframes_used']} keyframes, {cs['points_total']:,} points (showing {len(sub):,}); "
            f"median multi-view support {cs['median_support_among_2plus']:.0%}, "
            f"{cs['share_with_support_ge_0.7_among_2plus']:.0%} of points confirmed by >=70% of views")
    k0 = sorted(fr)[0]
    f0 = fr[k0]
    R0, C0 = np.array(f0["R_wc"]), np.array(f0["t_wc"])
    Z0 = np.load(paths.depth__aligned / f"{paths.frame_name(k0)}.npy")
    h, w = Z0.shape
    zmed = float(np.nanmedian(Z0))
    corners = [np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0]) * zmed * 0.25
               for u, v in ((0, 0), (w, 0), (w, h), (0, h))]
    graph_path = paths.tracking / "edge_graph.json"
    nodes, loops = [], []
    if graph_path.exists():
        G = read_json(graph_path)
        nodes = [np.round(disp(n["pos"]), 4).tolist() + [n["degree"]] for n in G["nodes"]]
        loops = [{"ring": np.round(disp(l["ring"]), 4).tolist(), "cls": l["support"]["class"]} for l in G["loops"]]
    quads = []
    surf_path = paths.tracking / "surfaces.json"
    if surf_path.exists():
        quads = [{"c": np.round(disp(np.asarray(q["c"])), 4).tolist(), "col": q["col"]}
                 for q in read_json(surf_path)["quads"]]
    data = {"pts": np.concatenate(pts).tolist(), "edges": E, "counts": counts, "kinds": kinds, "nodes": nodes,
            "loops": loops, "quads": quads, "nframes": len(fr), "note": note, "scale": zmed,
            "fov": float(2 * np.arctan(h / 2 / K[1, 1])),
            "cam": {"eye": disp(C0).tolist(), "target": disp(C0 + R0[:, 2] * zmed).tolist(),
                    "frustum": [disp(R0 @ c + C0).tolist() for c in corners]},
            "path": [disp(np.array(fr[k]["t_wc"])).tolist() for k in sorted(fr)], "center": disp(np.median(allw, 0)).tolist()}
    thumb = cv2.resize(cv2.imread(str(paths.keyframes / f"{paths.frame_name(k0)}.png")), (225, 400), interpolation=cv2.INTER_AREA)
    jpg = base64.b64encode(cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode()
    html = (HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__IMG__", jpg)
            .replace("__FRAME__", "map (all frames)").replace("provisional 3D edges", "persistent edge map"))
    out = paths.viz / "viewer_map.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {len(E)} map edges")
    return str(out)


def build(frame: int) -> str:
    cfg = load_config(CONFIG)
    paths = Paths(cfg)
    nm = paths.frame_name(frame)
    poses = read_json(paths.tracking / "camera_poses.json")
    fr = {f["frame"]: f for f in poses["frames"]}
    cam = next(iter(poses["cameras"].values()))
    K = np.array(cam["K"])
    f = fr[frame]
    R_wc, C = np.array(f["R_wc"]), np.array(f["t_wc"])

    disp = lambda P: np.asarray(P) * np.array([1.0, -1.0, 1.0])  # y down -> y up for display  # noqa: E731
    img = cv2.imread(str(paths.keyframes / f"{nm}.png"))
    Z = np.load(paths.depth__aligned / f"{nm}.npy")
    h, w = Z.shape
    step = 8
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    z = Z[ys, xs]
    ok = np.isfinite(z) & (z < 3 * np.nanmedian(Z))
    Xc = np.stack([(xs[ok] - K[0, 2]) / K[0, 0] * z[ok], (ys[ok] - K[1, 2]) / K[1, 1] * z[ok], z[ok]], 1)
    Xw = Xc @ R_wc.T + C
    rgb = img[ys[ok], xs[ok]][:, ::-1]
    pts = np.concatenate([np.round(disp(Xw), 4), rgb, np.full((len(Xw), 1), frame), np.full((len(Xw), 1), -1)], 1).ravel().tolist()

    edges = [e for e in read_json(paths.tracking / "edge_map_provisional.json")["edges"] if e["source"]["frame"] == frame]
    E = [{"t": e["type"], "k": e["edge_kind"], "c": e["confidence"],
          "v": np.round(disp(e["vertices_3d"]), 4).tolist()} for e in edges]
    counts = {"shapes": {}, "kinds": {}}
    for e in edges:
        counts["shapes"][e["type"]] = counts["shapes"].get(e["type"], 0) + 1
        counts["kinds"][e["edge_kind"]] = counts["kinds"].get(e["edge_kind"], 0) + 1

    zmed = float(np.nanmedian(Z))
    fwd = R_wc[:, 2]
    corners = [np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0]) * zmed * 0.25
               for u, v in ((0, 0), (w, 0), (w, h), (0, h))]
    frustum = [disp(R_wc @ c + C).tolist() for c in corners]
    center = disp(Xw.mean(0)).tolist()
    data = {"pts": pts, "edges": E, "counts": counts, "kinds": ["depth_discontinuity", "crease_candidate", "texture", "unknown"],
            "nodes": [], "loops": [], "quads": [], "nframes": 1, "note": "single keyframe's depth only", "scale": zmed, "fov": float(2 * np.arctan(h / 2 / K[1, 1])),
            "cam": {"eye": disp(C).tolist(), "target": disp(C + fwd * zmed).tolist(), "frustum": frustum},
            "path": [disp(np.array(fr[k]["t_wc"])).tolist() for k in sorted(fr)], "center": center}
    thumb = cv2.resize(img, (225, 400), interpolation=cv2.INTER_AREA)
    jpg = base64.b64encode(cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode()
    html = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__IMG__", jpg).replace("__FRAME__", str(frame))
    out = paths.viz / f"viewer_frame{frame:04d}.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {len(E)} edges, {len(pts) // 6} depth points")
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=13)
    ap.add_argument("--map", action="store_true", help="view the persistent map from stage 8 instead of one frame")
    ap.add_argument("--config", default="pipeline.yaml", help="pipeline config of the run to view (e.g. pipeline_mouse.yaml)")
    args = ap.parse_args()
    CONFIG = args.config
    build_map() if args.map else build(args.frame)
