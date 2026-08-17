"""Build a self-contained rotatable 3D HTML viewer for GT boxes.

Reads the JSON produced by extract_gtbox_sample.py and emits a single HTML file
that embeds the data and renders oriented 3D boxes with Three.js (OrbitControls).
No server / no external data file needed - just open the HTML in a browser.
"""
import argparse
import json

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>GT 3D Boxes - __TITLE__</title>
<style>
  html,body{margin:0;height:100%;background:#0d1117;color:#e6edf3;font-family:system-ui,Arial,sans-serif;overflow:hidden}
  #info{position:absolute;top:10px;left:10px;padding:10px 14px;background:rgba(22,27,34,.85);border:1px solid #30363d;border-radius:8px;font-size:13px;line-height:1.6;max-width:320px}
  #info b{color:#58a6ff}
  #legend{position:absolute;top:10px;right:10px;padding:10px 14px;background:rgba(22,27,34,.85);border:1px solid #30363d;border-radius:8px;font-size:13px;line-height:1.8}
  .sw{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:middle}
  #tip{position:absolute;padding:6px 9px;background:rgba(0,0,0,.85);border:1px solid #58a6ff;border-radius:6px;font-size:12px;pointer-events:none;display:none;white-space:nowrap}
  #hint{position:absolute;bottom:10px;left:10px;font-size:12px;color:#8b949e}
</style>
</head>
<body>
<div id="info"></div>
<div id="legend">
  <div><span class="sw" style="background:#3fb950"></span>静态 (|v| ≤ __VTHRESH__ m/s)</div>
  <div><span class="sw" style="background:#f85149"></span>动态 (|v| > __VTHRESH__ m/s)</div>
  <div><span class="sw" style="background:#f0d000"></span>自车 (ego, 原点)</div>
  <div><span class="sw" style="background:#f85149;height:2px;border-radius:0"></span>速度矢量</div>
  <hr style="border-color:#30363d;margin:6px 0">
  <div style="font-size:12px;color:#8b949e">Occ 语义 (点云)</div>
  <div id="occlegend"></div>
</div>
<div id="tip"></div>
<div id="hint">鼠标左键旋转 · 右键平移 · 滚轮缩放 · 悬停查看box信息<br>快捷键: <b>O</b> 切occ · <b>B</b> 切box · <b>[</b>/<b>]</b> occ点大小</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const DATA = __DATA__;

// nuScenes / SurroundOcc 17-class palette (label 0..16), RGB 0-255
const OCC_PALETTE = [
  [ 90, 90, 90], [255,120, 50], [255,192,203], [255,255,  0], [  0,150,245],
  [  0,255,255], [200,180,  0], [255,  0,  0], [255,240,150], [135, 60,  0],
  [160, 32,240], [255,  0,255], [139,137,137], [ 75,  0, 75], [150,240, 80],
  [230,230,250], [  0,175,  0],
];
const OCC_NAMES = ['others','barrier','bicycle','bus','car','construction_veh',
  'motorcycle','pedestrian','traffic_cone','trailer','truck','driveable_surface',
  'other_flat','sidewalk','terrain','manmade','vegetation'];

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
const camera = new THREE.PerspectiveCamera(55, window.innerWidth/window.innerHeight, 0.1, 2000);
camera.up.set(0,0,1);              // Z is up (LIDAR frame)
camera.position.set(45,-45,40);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.9));
const dl = new THREE.DirectionalLight(0xffffff, 0.6); dl.position.set(1,1,2); scene.add(dl);

// ---- ground grid (BEV plane, pc_range) ----
const [x0,y0,z0,x1,y1,z1] = DATA.pc_range;
const grid = new THREE.GridHelper(Math.max(x1-x0,y1-y0), 20, 0x30363d, 0x21262d);
grid.rotation.x = Math.PI/2;        // put grid on XY plane
grid.position.z = z0;
scene.add(grid);
// axes at ego
const axes = new THREE.AxesHelper(4); scene.add(axes);
// ego marker (yellow box ~ car footprint 4.08 x 1.73)
const ego = new THREE.Mesh(new THREE.BoxGeometry(4.08,1.73,1.5),
  new THREE.MeshBasicMaterial({color:0xf0d000, transparent:true, opacity:0.35}));
ego.position.z = 0.75; scene.add(ego);
const egoEdges = new THREE.LineSegments(new THREE.EdgesGeometry(ego.geometry),
  new THREE.LineBasicMaterial({color:0xf0d000})); egoEdges.position.copy(ego.position); scene.add(egoEdges);

// ---- occupancy voxel point cloud ----
let occPoints = null;
let occSize = 0.35;
if (DATA.occ && DATA.occ.length){
  const n = DATA.occ.length;
  const pos = new Float32Array(n*3), col = new Float32Array(n*3);
  const used = new Set();
  for (let p=0;p<n;p++){
    const v = DATA.occ[p];
    pos[p*3]=v[0]; pos[p*3+1]=v[1]; pos[p*3+2]=v[2];
    const c = OCC_PALETTE[v[3]] || [128,128,128];
    col[p*3]=c[0]/255; col[p*3+1]=c[1]/255; col[p*3+2]=c[2]/255;
    used.add(v[3]);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos,3));
  g.setAttribute('color', new THREE.BufferAttribute(col,3));
  occPoints = new THREE.Points(g, new THREE.PointsMaterial({
    size:occSize, vertexColors:true, sizeAttenuation:true}));
  scene.add(occPoints);
  // build occ legend for classes actually present
  const ll = [...used].sort((a,b)=>a-b).map(l=>{
    const c = OCC_PALETTE[l]||[128,128,128];
    return `<div><span class="sw" style="background:rgb(${c[0]},${c[1]},${c[2]})"></span>${OCC_NAMES[l]}</div>`;
  }).join('');
  document.getElementById('occlegend').innerHTML = ll;
}

// ---- boxes ----
const pickables = [];
const boxGroup = new THREE.Group(); scene.add(boxGroup);
DATA.boxes.forEach((b,i)=>{
  const col = b.dynamic ? 0xf85149 : 0x3fb950;
  const geo = new THREE.BoxGeometry(b.dx, b.dy, b.dz);
  const mesh = new THREE.Mesh(geo,
    new THREE.MeshBasicMaterial({color:col, transparent:true, opacity:0.12}));
  mesh.position.set(b.x, b.y, b.z);
  mesh.rotation.z = b.heading;
  boxGroup.add(mesh);
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo),
    new THREE.LineBasicMaterial({color:col}));
  edges.position.copy(mesh.position); edges.rotation.copy(mesh.rotation);
  boxGroup.add(edges);
  mesh.userData = {idx:i, box:b};
  pickables.push(mesh);
  // heading arrow (forward = +x in box local frame)
  const hx = Math.cos(b.heading), hy = Math.sin(b.heading);
  const half = b.dx/2 + 0.6;
  const hdir = new THREE.Vector3(hx,hy,0);
  const harrow = new THREE.ArrowHelper(hdir, new THREE.Vector3(b.x,b.y,b.z), half, col, 0.5, 0.3);
  boxGroup.add(harrow);
  // velocity vector (red) for dynamic boxes
  if (b.dynamic){
    const v = new THREE.Vector3(b.vx, b.vy, 0);
    const len = Math.min(v.length()*1.2, 12) + 0.5;
    const arrow = new THREE.ArrowHelper(v.clone().normalize(),
      new THREE.Vector3(b.x,b.y,b.z), len, 0xf85149, 1.0, 0.6);
    boxGroup.add(arrow);
  }
});

// ---- info panel ----
document.getElementById('info').innerHTML =
  `<b>GT 3D Boxes</b><br>`+
  `sample idx: ${DATA.index}<br>`+
  `token: ${DATA.token.slice(0,12)}…<br>`+
  `box 总数: <b>${DATA.n_boxes}</b><br>`+
  `动态 box: <b style="color:#f85149">${DATA.n_dynamic}</b> · 静态: <b style="color:#3fb950">${DATA.n_boxes-DATA.n_dynamic}</b><br>`+
  `occ 体素: <b>${(DATA.occ||[]).length}</b><br>`+
  `坐标系: LIDAR (Z 朝上)`;

// ---- keyboard toggles ----
window.addEventListener('keydown', e=>{
  const k = e.key.toLowerCase();
  if (k==='o' && occPoints) occPoints.visible = !occPoints.visible;
  else if (k==='b') boxGroup.visible = !boxGroup.visible;
  else if ((k===']'||k==='=') && occPoints){ occSize=Math.min(occSize+0.1,2); occPoints.material.size=occSize; }
  else if ((k==='['||k==='-') && occPoints){ occSize=Math.max(occSize-0.1,0.05); occPoints.material.size=occSize; }
});

// ---- hover pick ----
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
const tip = document.getElementById('tip');
renderer.domElement.addEventListener('mousemove', e=>{
  mouse.x = (e.clientX/window.innerWidth)*2-1;
  mouse.y = -(e.clientY/window.innerHeight)*2+1;
  ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObjects(pickables, false);
  if (hit.length){
    const b = hit[0].object.userData.box;
    tip.style.display='block';
    tip.style.left=(e.clientX+12)+'px'; tip.style.top=(e.clientY+12)+'px';
    tip.innerHTML = `<b>${b.name}</b> ${b.dynamic?'<span style="color:#f85149">[动]</span>':'<span style="color:#3fb950">[静]</span>'}<br>`+
      `pos (${b.x.toFixed(1)}, ${b.y.toFixed(1)}, ${b.z.toFixed(1)})<br>`+
      `size ${b.dx.toFixed(2)}×${b.dy.toFixed(2)}×${b.dz.toFixed(2)}<br>`+
      `heading ${b.heading.toFixed(2)} rad · |v| ${b.speed.toFixed(2)} m/s`;
  } else { tip.style.display='none'; }
});

window.addEventListener('resize', ()=>{
  camera.aspect = window.innerWidth/window.innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
(function loop(){ requestAnimationFrame(loop); controls.update(); renderer.render(scene,camera); })();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    with open(args.json) as f:
        data = json.load(f)
    html = (HTML_TEMPLATE
            .replace('__DATA__', json.dumps(data))
            .replace('__TITLE__', str(data.get('token', ''))[:12])
            .replace('__VTHRESH__', str(data.get('v_thresh', 0.5))))
    with open(args.out, 'w') as f:
        f.write(html)
    print(f'wrote {args.out} ({len(html)} bytes, {data["n_boxes"]} boxes)')


if __name__ == '__main__':
    main()
