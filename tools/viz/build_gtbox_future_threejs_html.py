"""Build a Three.js GT-box future-track viewer in the gtbox_sample_3d style.

The rendered scene intentionally matches ``gtbox_sample_3d.html``: dark UI,
semantic SurroundOcc point cloud, colored transparent GT box volumes, edges,
heading arrows, OrbitControls and hover tooltips.  It adds a time slider and
animation: selected dynamic GT boxes move through their real future GT-box
centers, while glowing colored trajectory lines remain visible.
"""
import argparse
import json
import os
import pickle

import numpy as np


DT, STEPS = 0.5, 6
MOVABLE = {'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
           'motorcycle', 'bicycle', 'pedestrian'}
COLORS = [0xff453a, 0x0a84ff, 0x30d158, 0xff9f0a, 0xbf5af2, 0x64d2ff,
          0xff375f, 0x5e5ce6]


def load_infos(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
    return [frame for frames in infos.values() for frame in frames] if isinstance(infos, dict) else infos


def get_track_arrays(info):
    boxes = np.asarray(info['gt_boxes'], dtype=np.float32)
    names = np.asarray(info.get('gt_names', ['unknown'] * len(boxes))).astype(str)
    velocity = np.nan_to_num(np.asarray(info.get('gt_velocity', np.zeros((len(boxes), 2))), np.float32))
    delta = np.nan_to_num(np.asarray(info['gt_agent_fut_trajs'], np.float32)).reshape(-1, STEPS, 2)
    mask = np.nan_to_num(np.asarray(info['gt_agent_fut_masks'], np.float32)).reshape(-1, STEPS) > .5
    yaw_delta = np.nan_to_num(np.asarray(
        info.get('gt_agent_fut_yaw', np.zeros((len(boxes), STEPS))), np.float32)).reshape(-1, STEPS)
    cum = np.cumsum(delta, axis=1)
    cum_yaw = np.cumsum(yaw_delta, axis=1)
    extent = np.where(mask, np.linalg.norm(cum, axis=-1), 0).max(axis=1)
    movable = np.array([name in MOVABLE for name in names])
    return boxes, names, velocity, cum, cum_yaw, mask, extent, movable


def auto_index(infos, max_tracks, min_motion):
    best, best_score = 0, -1.0
    for i, info in enumerate(infos):
        if 'gt_agent_fut_trajs' not in info:
            continue
        boxes, _, _, _, _, mask, extent, movable = get_track_arrays(info)
        if not len(boxes):
            continue
        selected = movable & mask.any(1) & (extent >= min_motion)
        score = min(int(selected.sum()), max_tracks) * 100 + float(extent[selected].sum())
        if score > best_score:
            best, best_score = i, score
    return best


def load_occ(info, data_root, max_points):
    path = str(info.get('occ_path', ''))
    path = path if os.path.isabs(path) else os.path.join(data_root, path)
    if not path or not os.path.exists(path):
        return []
    occ = np.load(path)
    i, j, k, label = occ[:, 0], occ[:, 1], occ[:, 2], occ[:, 3]
    keep = ((i >= 40) & (i < 160) & (j >= 40) & (j < 160) &
            (k >= 6) & (k < 14) & (label != 17))
    rows = np.stack([(i[keep] - 40) * .5 + .25 - 30.,
                     (j[keep] - 40) * .5 + .25 - 30.,
                     (k[keep] - 6) * .5 + .25 - 2., label[keep]], axis=1)
    if len(rows) > max_points:
        rows = rows[np.random.RandomState(0).choice(len(rows), max_points, replace=False)]
    return rows.astype(np.float32).round(3).tolist()


def build_data(info, index, data_root, max_tracks, min_motion, max_occ):
    boxes, names, velocity, cum, cum_yaw, mask, extent, movable = get_track_arrays(info)
    speeds = np.linalg.norm(velocity, axis=1)
    selected = np.flatnonzero(movable & mask.any(1) & (extent >= min_motion))
    selected = selected[np.argsort(extent[selected])[::-1][:max_tracks]]
    selected_set = set(selected.tolist())
    static_boxes, tracks = [], []
    for i, box in enumerate(boxes):
        item = dict(idx=int(i), x=float(box[0]), y=float(box[1]), z=float(box[2]),
                    dx=float(box[3]), dy=float(box[4]), dz=float(box[5]), heading=float(box[6]),
                    vx=float(velocity[i, 0]), vy=float(velocity[i, 1]), speed=float(speeds[i]),
                    dynamic=bool(speeds[i] > .5), name=str(names[i]))
        if i not in selected_set:
            static_boxes.append(item)
    for local, i in enumerate(selected):
        box = boxes[i]
        frames = [dict(x=float(box[0]), y=float(box[1]), z=float(box[2]), yaw=float(box[6]))]
        for t in range(STEPS):
            frames.append(dict(x=float(box[0] + cum[i, t, 0]), y=float(box[1] + cum[i, t, 1]),
                               z=float(box[2]), yaw=float(box[6] + cum_yaw[i, t])))
        tracks.append(dict(idx=int(i), name=str(names[i]), color=int(COLORS[local % len(COLORS)]),
                           dx=float(box[3]), dy=float(box[4]), dz=float(box[5]),
                           speed=float(speeds[i]), extent=float(extent[i]), frames=frames))
    return dict(index=index, token=str(info.get('token', '')), scene=str(info.get('scene_name', '')),
                pc_range=[-30, -30, -2, 30, 30, 2], occ=load_occ(info, data_root, max_occ),
                boxes=static_boxes, tracks=tracks, n_boxes=int(len(boxes)))


TEMPLATE = r'''<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"/>
<title>GT Box Future Tracks</title><style>
html,body{margin:0;height:100%;background:#0d1117;color:#e6edf3;font-family:system-ui,Arial,sans-serif;overflow:hidden}
#info,#legend,#controls{position:absolute;padding:10px 14px;background:rgba(22,27,34,.88);border:1px solid #30363d;border-radius:8px;font-size:13px;line-height:1.6;z-index:2}
#info{top:10px;left:10px;max-width:365px}#legend{top:10px;right:10px}#controls{bottom:10px;left:10px;right:10px}
#timeline{width:calc(100% - 180px);vertical-align:middle}.sw{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:middle}
#tip{position:absolute;padding:6px 9px;background:rgba(0,0,0,.88);border:1px solid #58a6ff;border-radius:6px;font-size:12px;pointer-events:none;display:none;white-space:nowrap;z-index:3}
button{background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:5px;padding:4px 12px;margin-right:8px;cursor:pointer}button:hover{background:#30363d}
</style></head><body><div id="info"></div><div id="legend"><div><span class="sw" style="background:#3fb950"></span>静态 GT box</div><div><span class="sw" style="background:#f85149"></span>动态 GT box（未追踪）</div><div><span class="sw" style="background:#58a6ff"></span>未来轨迹 GT box</div><div><span class="sw" style="background:#f0d000"></span>自车 / LiDAR 原点</div><hr style="border-color:#30363d"><div>O：切 Occ　B：切静态 box</div></div><div id="controls"><button id="play">▶ 播放</button><button id="reset">↺ t=0</button><b id="time">t=0（当前）</b>　<input id="timeline" type="range" min="0" max="6" value="0" step="1"/></div><div id="tip"></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script><script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script><script>
const DATA=__DATA__;const PAL=[[90,90,90],[255,120,50],[255,192,203],[255,255,0],[0,150,245],[0,255,255],[200,180,0],[255,0,0],[255,240,150],[135,60,0],[160,32,240],[255,0,255],[139,137,137],[75,0,75],[150,240,80],[230,230,250],[0,175,0]];
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0d1117);const camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,.1,2000);camera.up.set(0,0,1);camera.position.set(45,-45,40);const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(innerWidth,innerHeight);renderer.setPixelRatio(devicePixelRatio);document.body.appendChild(renderer.domElement);const orbit=new THREE.OrbitControls(camera,renderer.domElement);orbit.enableDamping=true;scene.add(new THREE.AmbientLight(0xffffff,.9));const light=new THREE.DirectionalLight(0xffffff,.6);light.position.set(1,1,2);scene.add(light);
const grid=new THREE.GridHelper(60,20,0x30363d,0x21262d);grid.rotation.x=Math.PI/2;grid.position.z=-2;scene.add(grid);scene.add(new THREE.AxesHelper(4));
const ego=new THREE.Mesh(new THREE.BoxGeometry(4.08,1.73,1.5),new THREE.MeshBasicMaterial({color:0xf0d000,transparent:true,opacity:.35}));ego.position.z=.75;scene.add(ego);const egoE=new THREE.LineSegments(new THREE.EdgesGeometry(ego.geometry),new THREE.LineBasicMaterial({color:0xf0d000}));egoE.position.copy(ego.position);scene.add(egoE);
let occ=null;if(DATA.occ.length){const p=new Float32Array(DATA.occ.length*3),c=new Float32Array(DATA.occ.length*3);DATA.occ.forEach((v,i)=>{p.set(v.slice(0,3),i*3);const q=PAL[v[3]]||[128,128,128];c.set(q.map(x=>x/255),i*3)});const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(p,3));g.setAttribute('color',new THREE.BufferAttribute(c,3));occ=new THREE.Points(g,new THREE.PointsMaterial({size:.35,vertexColors:true,sizeAttenuation:true}));scene.add(occ)}
const staticGroup=new THREE.Group(),trackGroup=new THREE.Group();scene.add(staticGroup);scene.add(trackGroup);const pick=[];
function colorHex(n){return n}function makeBox(b,color,opacity=.12){const geo=new THREE.BoxGeometry(b.dx,b.dy,b.dz),mesh=new THREE.Mesh(geo,new THREE.MeshBasicMaterial({color,transparent:true,opacity}));mesh.position.set(b.x,b.y,b.z);mesh.rotation.z=b.heading||b.yaw||0;const ed=new THREE.LineSegments(new THREE.EdgesGeometry(geo),new THREE.LineBasicMaterial({color}));ed.position.copy(mesh.position);ed.rotation.copy(mesh.rotation);const group=new THREE.Group();group.add(mesh);group.add(ed);return {group,mesh}}
DATA.boxes.forEach(b=>{const col=b.dynamic?0xf85149:0x3fb950;const o=makeBox(b,col);o.mesh.userData={box:b};pick.push(o.mesh);staticGroup.add(o.group)});
const animated=[];DATA.tracks.forEach((t,i)=>{const first=Object.assign({},t.frames[0],t);const o=makeBox(first,t.color,.22);o.mesh.userData={box:t};pick.push(o.mesh);trackGroup.add(o.group);animated.push({t,o});const pts=t.frames.map(f=>new THREE.Vector3(f.x,f.y,f.z+t.dz/2+.25));const glow=new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:t.color,transparent:true,opacity:.25}));trackGroup.add(glow);const ray=new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:t.color}));trackGroup.add(ray);const arr=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),pts[0],.8,t.color,.3,.2);trackGroup.add(arr)});
function update(step){animated.forEach(({t,o})=>{const f=t.frames[step];o.group.position.set(f.x-t.frames[0].x,f.y-t.frames[0].y,0);o.group.rotation.z=f.yaw-t.frames[0].yaw});document.getElementById('timeline').value=step;document.getElementById('time').textContent=step?`t=${step}（+${(step*.5).toFixed(1)}s）`:'t=0（当前）'}
let timer=null;document.getElementById('timeline').oninput=e=>update(+e.target.value);document.getElementById('reset').onclick=()=>update(0);document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='▶ 播放';return}let s=+document.getElementById('timeline').value;timer=setInterval(()=>{s=(s+1)%7;update(s)},700);document.getElementById('play').textContent='⏸ 暂停'};
document.getElementById('info').innerHTML=`<b>GT Box 真实未来轨迹</b><br>sample idx: ${DATA.index}<br>token: ${DATA.token.slice(0,12)}…<br>当前 GT box: <b>${DATA.n_boxes}</b><br>展示轨迹实例: <b>${DATA.tracks.length}</b><br>Occ 语义体素: <b>${DATA.occ.length}</b><br><span style="color:#8b949e">彩色实体 box 沿同一实例的未来 GT box 移动；同色光线是其中心轨迹。</span>`;
addEventListener('keydown',e=>{if(e.key.toLowerCase()==='o'&&occ)occ.visible=!occ.visible;if(e.key.toLowerCase()==='b')staticGroup.visible=!staticGroup.visible});const raycaster=new THREE.Raycaster(),mouse=new THREE.Vector2(),tip=document.getElementById('tip');renderer.domElement.addEventListener('mousemove',e=>{mouse.x=e.clientX/innerWidth*2-1;mouse.y=-(e.clientY/innerHeight)*2+1;raycaster.setFromCamera(mouse,camera);const h=raycaster.intersectObjects(pick,false);if(h.length){const b=h[0].object.userData.box;tip.style.display='block';tip.style.left=e.clientX+12+'px';tip.style.top=e.clientY+12+'px';tip.innerHTML=`<b>${b.name}</b><br>GT instance #${b.idx}<br>速度 ${b.speed.toFixed(2)} m/s${b.extent?`<br>3秒轨迹 ${b.extent.toFixed(1)} m`:''}`}else tip.style.display='none'});addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});(function loop(){requestAnimationFrame(loop);orbit.update();renderer.render(scene,camera)})();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', required=True)
    parser.add_argument('--data-root', default='.')
    parser.add_argument('--out', required=True)
    parser.add_argument('--index', type=int, default=-1)
    parser.add_argument('--max-tracks', type=int, default=6)
    parser.add_argument('--min-motion', type=float, default=1.)
    parser.add_argument('--max-occ', type=int, default=60000)
    args = parser.parse_args()
    infos = load_infos(args.pkl)
    index = args.index if args.index >= 0 else auto_index(infos, args.max_tracks, args.min_motion)
    data = build_data(infos[index], index, args.data_root, args.max_tracks, args.min_motion, args.max_occ)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as handle:
        handle.write(TEMPLATE.replace('__DATA__', json.dumps(data, separators=(',', ':'))))
    print(f'wrote {args.out}: sample={index}, tracks={len(data["tracks"])}, occ={len(data["occ"])}')


if __name__ == '__main__':
    main()