// whittle GUI: load a part, place supports and loads (or pick the wind), run a
// sweep, slide through it. All the physics happens in the Python server; this
// page draws, picks, and asks.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const G = 9.81;

// Colour ramps, from the data-viz reference palette. Stress is a magnitude:
// one hue, light to dark. Drag is signed: blue (pulls forward) through a
// neutral grey to red (pushes back).
const SEQUENTIAL = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
const DIVERGING = ['#104281', '#2a78d6', '#86b6ef', '#f0efec', '#f0a3a2', '#e34948', '#a8312f'];
const COLORS = { support: '#2a78d6', load: '#eb6834', keep: '#1baf7a' };
const PART_COLOR = 0xb4bfcb;

const state = {
  info: null,
  part: null,
  file: null,
  kind: 'load',
  tool: null,
  select: 'face',
  flow: [-1, 0, 0],
  items: [],
  job: null,
  result: null,
  station: 0,
  geometries: new Map(),
  stale: false,
  poll: null,
};

// ---------------------------------------------------------------- server

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let body = {};
  try { body = await response.json(); } catch { /* an empty or non-JSON reply */ }
  if (!response.ok) throw new Error(body.error || `${response.status} ${response.statusText}`);
  return body;
}

const post = (path, body) => api(path, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

function decode(base64, Type) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Type(bytes.buffer);
}

// ---------------------------------------------------------------- scene

const viewport = $('viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
viewport.prepend(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 1e5);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;

scene.add(new THREE.HemisphereLight(0xffffff, 0x9a988f, 1.5));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.7);
keyLight.position.set(0.6, 0.5, 1);
camera.add(keyLight);
scene.add(camera);

const layers = {
  original: new THREE.Group(),
  station: new THREE.Group(),
  ghost: new THREE.Group(),
  markers: new THREE.Group(),
  flow: new THREE.Group(),
  helpers: new THREE.Group(),
};
Object.values(layers).forEach((group) => scene.add(group));

const partMaterial = new THREE.MeshStandardMaterial({ color: PART_COLOR, roughness: 0.62, metalness: 0.04 });
const heatMaterial = new THREE.MeshStandardMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.7, metalness: 0.0 });
const ghostMaterial = new THREE.MeshBasicMaterial({
  color: 0x7d8896, transparent: true, opacity: 0.1, depthWrite: false,
  polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 2,
});

let originalMesh = null;

function resize() {
  const { clientWidth: w, clientHeight: h } = viewport;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

// Station geometries are cached for the slider, so clearing that layer must
// not dispose them; everything else is drawn fresh and can go.
function clear(group, dispose = true) {
  for (const child of [...group.children]) {
    group.remove(child);
    if (!dispose) continue;
    child.traverse?.((node) => {
      node.geometry?.dispose?.();
      if (node.material && node.material !== partMaterial && node.material !== heatMaterial && node.material !== ghostMaterial) {
        node.material.dispose?.();
      }
    });
  }
}

function geometryFrom(payload) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(decode(payload.positions, Float32Array), 3));
  geometry.setIndex(new THREE.BufferAttribute(decode(payload.indices, Uint32Array), 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  if (payload.heat) geometry.userData.heat = decode(payload.heat, Float32Array);
  return geometry;
}

function bounds() {
  const min = new THREE.Vector3(...state.part.min);
  const max = new THREE.Vector3(...state.part.max);
  return { min, max, center: min.clone().add(max).multiplyScalar(0.5), size: max.clone().sub(min) };
}

function frame() {
  const { center, size } = bounds();
  const radius = Math.max(size.length() / 2, 1e-3);
  const direction = new THREE.Vector3(0.9, -1.5, 0.85).normalize();
  // Airflow mode draws wind arrows upstream of the part; leave room for them.
  const distance = radius * (state.kind === 'flow' ? 3.5 : 2.8);
  camera.position.copy(center).addScaledVector(direction, distance);
  camera.near = radius / 200;
  camera.far = radius * 200;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

function drawHelpers() {
  clear(layers.helpers);
  const { min, center, size } = bounds();
  const extent = Math.max(size.x, size.y, size.z);
  const step = niceStep(extent / 6);
  const span = Math.ceil((extent * 2.2) / step) * step;
  const grid = new THREE.GridHelper(span, Math.round(span / step), 0x9d9b93, 0xc9c7bf);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(center.x, center.y, min.z - extent * 0.002);
  grid.material.transparent = true;
  grid.material.opacity = 0.55;
  layers.helpers.add(grid);
  const axes = new THREE.AxesHelper(extent * 0.18);
  axes.position.copy(min);
  layers.helpers.add(axes);
  layers.helpers.userData.step = step;
}

function niceStep(raw) {
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / power;
  return (unit < 1.5 ? 1 : unit < 3.5 ? 2 : unit < 7.5 ? 5 : 10) * power;
}

// ---------------------------------------------------------------- heat

function rampColor(stops, t) {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const a = new THREE.Color(stops[i]);
  const b = new THREE.Color(stops[i + 1]);
  return a.lerp(b, x - i);
}

function colorize(geometry) {
  const heat = geometry.userData.heat;
  const result = state.result;
  if (!heat || !result) return false;
  const signed = result.heat.signed;
  const top = result.heatMax || 1;
  const colors = new Float32Array(heat.length * 3);
  for (let i = 0; i < heat.length; i++) {
    const t = signed ? 0.5 + 0.5 * (heat[i] / top) : heat[i] / top;
    const c = rampColor(signed ? DIVERGING : SEQUENTIAL, t);
    colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  return true;
}

function drawLegend() {
  const result = state.result;
  const show = result && $('heat-toggle').checked && !state.tool;
  $('legend').hidden = !show;
  if (!show) return;
  const signed = result.heat.signed;
  const stops = signed ? DIVERGING : SEQUENTIAL;
  $('legend-bar').style.background = `linear-gradient(to right, ${stops.join(', ')})`;
  const top = result.heatMax;
  if (signed) {
    $('legend-title').textContent = 'Drag per area: blue pulls forward, red pushes back';
    $('legend-min').textContent = `−${top.toFixed(2)}`;
    $('legend-mid').textContent = '0';
    $('legend-max').textContent = `+${top.toFixed(2)}`;
  } else {
    $('legend-title').textContent = `Stress, MPa (material breaks near ${result.strength})`;
    $('legend-min').textContent = '0';
    $('legend-mid').textContent = (top / 2).toFixed(1);
    $('legend-max').textContent = top.toFixed(1);
  }
}

// ---------------------------------------------------------------- parts

async function loadInfo() {
  state.info = await api('/api/info');
  const examples = $('example-select');
  for (const example of state.info.examples) {
    const option = document.createElement('option');
    option.value = example.id;
    option.textContent = `${example.name} (${example.kind === 'load' ? 'strength' : 'airflow'})`;
    option.title = example.description;
    examples.append(option);
  }
  const materials = $('material-select');
  for (const [key, m] of Object.entries(state.info.materials)) {
    const option = document.createElement('option');
    option.value = key;
    option.textContent = `${m.name} — ${m.strength} MPa`;
    materials.append(option);
  }
}

async function openFile(file) {
  if (!file) return;
  state.file = file;
  setBusy(`Reading ${file.name}…`);
  try {
    const units = $('units-select').value;
    const payload = await api(`/api/parts?name=${encodeURIComponent(file.name)}&units=${units}`, {
      method: 'POST', body: await file.arrayBuffer(),
    });
    $('example-select').value = '';
    setPart(payload);
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy('');
  }
}

async function openExample(id) {
  if (!id) return;
  setBusy('Loading example…');
  try {
    const payload = await post(`/api/examples/${id}`, {});
    state.file = null;
    setPart(payload);
    await applySetup(payload.setup);
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy('');
  }
}

function setPart(payload) {
  stopPolling();
  state.part = payload;
  state.items = [];
  state.result = null;
  state.job = null;
  state.geometries.clear();
  clear(layers.markers);
  clear(layers.station, false);
  state.geometries.forEach((geometry) => geometry.dispose());
  clear(layers.ghost);
  clear(layers.original);
  const geometry = geometryFrom(payload.mesh);
  originalMesh = new THREE.Mesh(geometry, partMaterial);
  layers.original.add(originalMesh);
  layers.ghost.add(new THREE.Mesh(geometry, ghostMaterial));
  drawHelpers();
  frame();
  $('empty').hidden = true;
  $('results').hidden = true;
  $('part-title').textContent = payload.name;
  const s = bounds().size;
  const info = `${fmtLength(s.x)} × ${fmtLength(s.y)} × ${fmtLength(s.z)} mm · ${payload.triangles.toLocaleString()} triangles`;
  $('part-info').textContent = payload.closed ? info : `${info} — not watertight, so results may be wrong`;
  $('part-info').classList.toggle('warn', !payload.closed);
  prefillKeepBox();
  showError('');
  renderItems();
  drawFlow();
  updateView();
  updateRunButton();
}

const fmtLength = (v) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));

// ---------------------------------------------------------------- set-up

function setKind(kind) {
  state.kind = kind;
  $('tab-load').setAttribute('aria-selected', String(kind === 'load'));
  $('tab-flow').setAttribute('aria-selected', String(kind === 'flow'));
  $('load-panel').hidden = kind !== 'load';
  $('flow-panel').hidden = kind !== 'flow';
  setTool(null);
  const remove = kind === 'load' ? 60 : 25;
  $('remove-input').value = remove;
  $('stations-input').value = kind === 'load' ? 7 : 6;
  syncOutputs();
  drawFlow();
  if (state.part) frame();
  markStale();
  updateView();
  updateRunButton();
}

function setTool(tool) {
  state.tool = state.tool === tool ? null : tool;
  document.querySelectorAll('.tool').forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.tool === state.tool));
  });
  $('load-options').hidden = state.tool !== 'load';
  viewport.classList.toggle('placing', Boolean(state.tool));
  updateView();
}

function setSelect(mode) {
  state.select = mode;
  document.querySelectorAll('[data-select]').forEach((button) => {
    button.setAttribute('aria-checked', String(button.dataset.select === mode));
  });
  $('radius-field').hidden = mode !== 'point';
  updateHint();
}

function setFlow(vector) {
  state.flow = vector;
  document.querySelectorAll('[data-flow]').forEach((button) => {
    button.setAttribute('aria-checked', String(button.dataset.flow === vector.join(',')));
  });
  drawFlow();
  markStale();
}

function forceVector(normal) {
  const amount = Number($('force-input').value) || 0;
  const newtons = $('force-unit').value === 'kg' ? amount * G : amount;
  const mode = $('force-direction').value;
  let direction;
  if (mode === 'into') direction = new THREE.Vector3(...normal).negate();
  else if (mode === 'custom') direction = new THREE.Vector3(+$('dir-x').value, +$('dir-y').value, +$('dir-z').value);
  else direction = new THREE.Vector3(0, 0, -1);
  if (direction.lengthSq() === 0) direction.set(0, 0, -1);
  return direction.normalize().multiplyScalar(newtons).toArray();
}

async function place(hit) {
  const normal = hit.face.normal.clone().transformDirection(originalMesh.matrixWorld).toArray();
  const item = { type: state.tool, at: hit.point.toArray(), normal, mode: state.select };
  if (state.select === 'point') item.radius = Number($('radius-input').value) || 5;
  if (item.type === 'load') item.force = forceVector(normal);
  await addItem(item);
}

async function addItem(item) {
  state.items.push(item);
  try {
    await previewItem(item);
  } catch (error) {
    state.items.splice(state.items.indexOf(item), 1);
    showError(error.message);
  }
  renderItems();
  markStale();
  updateRunButton();
}

function itemRegion(item) {
  const region = { at: item.at };
  if (item.mode === 'face') region.normal = item.normal;
  if (item.radius != null) region.radius = item.radius;
  return region;
}

async function previewItem(item) {
  const preview = await post('/api/preview', {
    part: state.part.id, resolution: Number($('load-resolution').value), ...itemRegion(item),
  });
  removeObjects(item);
  item.count = preview.count;
  const points = new THREE.BufferGeometry();
  points.setAttribute('position', new THREE.BufferAttribute(decode(preview.points, Float32Array), 3));
  // Selected voxels sit just under the surface, so draw them see-through:
  // a support on the back face should still be visible from the front.
  const marker = new THREE.Points(points, new THREE.PointsMaterial({
    color: COLORS[item.type], size: preview.spacing * 0.9, sizeAttenuation: true,
    transparent: true, opacity: 0.55, depthTest: false,
  }));
  marker.renderOrder = 2;
  item.objects = [marker];
  if (item.type === 'load') {
    const force = new THREE.Vector3(...item.force);
    const length = Math.max(bounds().size.length() * 0.22, 1e-3);
    const direction = force.lengthSq() ? force.clone().normalize() : new THREE.Vector3(0, 0, -1);
    const origin = new THREE.Vector3(...item.at).addScaledVector(direction, -length);
    item.objects.push(new THREE.ArrowHelper(direction, origin, length, COLORS.load, length * 0.25, length * 0.12));
  }
  item.objects.forEach((object) => layers.markers.add(object));
}

function removeObjects(item) {
  for (const object of item.objects || []) {
    layers.markers.remove(object);
    object.traverse((node) => { node.geometry?.dispose?.(); node.material?.dispose?.(); });
  }
  item.objects = [];
}

function renderItems() {
  const list = $('placed-list');
  list.replaceChildren();
  for (const item of state.items) {
    const li = document.createElement('li');
    const dot = document.createElement('i');
    dot.className = `dot ${item.type}`;
    const label = document.createElement('span');
    label.className = 'label';
    const name = document.createElement('strong');
    name.textContent = { support: 'Support', load: 'Load', keep: 'Keep' }[item.type];
    label.append(name, document.createTextNode(` · ${describe(item)}`));
    const remove = document.createElement('button');
    remove.textContent = '×';
    remove.setAttribute('aria-label', `Remove ${name.textContent.toLowerCase()}`);
    remove.addEventListener('click', () => {
      removeObjects(item);
      state.items.splice(state.items.indexOf(item), 1);
      renderItems();
      markStale();
      updateRunButton();
    });
    li.append(dot, label, remove);
    list.append(li);
  }
}

function describe(item) {
  const where = item.mode === 'face' ? 'face' : `${item.radius} mm around point`;
  const voxels = item.count ? `, ${item.count} voxels` : '';
  if (item.type !== 'load') return `${where}${voxels}`;
  const newtons = Math.hypot(...item.force);
  return `${newtons.toFixed(0)} N (${(newtons / G).toFixed(1)} kg) on ${where}`;
}

async function refreshPreviews() {
  for (const item of state.items) {
    try { await previewItem(item); } catch (error) { showError(error.message); }
  }
  renderItems();
}

function drawFlow() {
  clear(layers.flow);
  if (!state.part || state.kind !== 'flow') return;
  const { center, size } = bounds();
  const flow = new THREE.Vector3(...state.flow).normalize();
  const extent = size.length();
  const along = Math.abs(size.dot(new THREE.Vector3(Math.abs(flow.x), Math.abs(flow.y), Math.abs(flow.z))));
  const helper = Math.abs(flow.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
  const u = new THREE.Vector3().crossVectors(flow, helper).normalize();
  const v = new THREE.Vector3().crossVectors(flow, u).normalize();
  const spanU = Math.abs(size.dot(new THREE.Vector3(Math.abs(u.x), Math.abs(u.y), Math.abs(u.z)))) * 0.35;
  const spanV = Math.abs(size.dot(new THREE.Vector3(Math.abs(v.x), Math.abs(v.y), Math.abs(v.z)))) * 0.35;
  const length = extent * 0.18;
  for (const a of [-1, 0, 1]) {
    for (const b of [-1, 0, 1]) {
      const origin = center.clone()
        .addScaledVector(flow, -(along / 2 + length * 1.15))
        .addScaledVector(u, a * spanU)
        .addScaledVector(v, b * spanV);
      layers.flow.add(new THREE.ArrowHelper(flow, origin, length, 0x8a96a6, length * 0.25, length * 0.12));
    }
  }
  if ($('keep-box-toggle').checked) {
    const box = keepBox();
    if (box) {
      const outline = new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3(...box[0]), new THREE.Vector3(...box[1])), COLORS.keep);
      // It sits inside the part; draw it through the surface.
      outline.material.depthTest = false;
      outline.material.transparent = true;
      outline.renderOrder = 2;
      layers.flow.add(outline);
    }
  }
}

function prefillKeepBox() {
  const { min, max, size } = bounds();
  const values = {
    'kb-x0': min.x + size.x * 0.15, 'kb-y0': min.y + size.y * 0.15, 'kb-z0': min.z,
    'kb-x1': max.x - size.x * 0.15, 'kb-y1': max.y - size.y * 0.15, 'kb-z1': min.z + size.z * 0.7,
  };
  for (const [id, value] of Object.entries(values)) $(id).value = Number(value.toPrecision(4));
}

function keepBox() {
  const read = (id) => Number($(id).value);
  const box = [[read('kb-x0'), read('kb-y0'), read('kb-z0')], [read('kb-x1'), read('kb-y1'), read('kb-z1')]];
  return box.flat().every(Number.isFinite) ? box : null;
}

async function applySetup(setup) {
  setKind(setup.kind);
  $('remove-input').value = Math.round(setup.removeUpTo * 100);
  $('stations-input').value = setup.stations;
  if (setup.kind === 'load') {
    $('material-select').value = setup.material;
    $('load-resolution').value = setup.resolution;
    syncOutputs();
    for (const s of setup.supports || []) {
      await addItem({ type: 'support', at: s.at, normal: s.normal, mode: s.normal ? 'face' : 'point', radius: s.radius });
    }
    for (const l of setup.loads || []) {
      await addItem({ type: 'load', at: l.at, normal: l.normal, mode: l.normal ? 'face' : 'point', radius: l.radius, force: l.force });
    }
  } else {
    setFlow(setup.flow);
    $('mirror-select').value = setup.mirror || '';
    $('extruded-select').value = setup.extruded || '';
    document.querySelectorAll('.flat-sides input').forEach((box) => { box.checked = (setup.flat || []).includes(box.value); });
    $('speed-input').value = setup.speedKmh;
    $('flow-resolution').value = setup.resolution;
    const keep = (setup.keepBoxes || [])[0];
    $('keep-box-toggle').checked = Boolean(keep);
    $('keep-box-fields').hidden = !keep;
    if (keep) {
      ['kb-x0', 'kb-y0', 'kb-z0'].forEach((id, i) => { $(id).value = keep[0][i]; });
      ['kb-x1', 'kb-y1', 'kb-z1'].forEach((id, i) => { $(id).value = keep[1][i]; });
    }
    syncOutputs();
    drawFlow();
  }
  state.stale = false;
  updateRunButton();
}

// ---------------------------------------------------------------- picking

let downAt = null;
renderer.domElement.addEventListener('pointerdown', (event) => { downAt = [event.clientX, event.clientY]; });
renderer.domElement.addEventListener('pointerup', (event) => {
  if (!downAt || !state.tool || !originalMesh) return;
  const moved = Math.hypot(event.clientX - downAt[0], event.clientY - downAt[1]);
  downAt = null;
  if (moved > 4) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const pointer = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  );
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(pointer, camera);
  const [hit] = raycaster.intersectObject(originalMesh);
  if (hit) place(hit);
});

// ---------------------------------------------------------------- running

function updateRunButton() {
  const ready = state.part && (state.kind === 'flow' ||
    (state.items.some((i) => i.type === 'support') && state.items.some((i) => i.type === 'load')));
  const running = state.job && ['queued', 'running'].includes(state.job.state);
  $('run-button').disabled = !ready || running;
  $('run-button').title = ready ? '' : (state.part ? 'Place at least one support and one load' : 'Load a part first');
}

function runBody() {
  const body = {
    kind: state.kind,
    part: state.part.id,
    removeUpTo: Number($('remove-input').value) / 100,
    stations: Number($('stations-input').value),
  };
  if (state.kind === 'load') {
    const of = (type) => state.items.filter((i) => i.type === type);
    Object.assign(body, {
      material: $('material-select').value,
      resolution: Number($('load-resolution').value),
      supports: of('support').map(itemRegion),
      loads: of('load').map((i) => ({ ...itemRegion(i), force: i.force })),
      keeps: of('keep').map(itemRegion),
    });
  } else {
    const box = $('keep-box-toggle').checked ? keepBox() : null;
    Object.assign(body, {
      flow: state.flow,
      resolution: Number($('flow-resolution').value),
      detail: 2,
      mirror: $('mirror-select').value || null,
      extruded: $('extruded-select').value || null,
      flat: [...document.querySelectorAll('.flat-sides input:checked')].map((i) => i.value),
      speedKmh: Number($('speed-input').value),
      effort: Number($('effort-select').value),
      keepBoxes: box ? [box] : [],
    });
  }
  return body;
}

async function run() {
  showError('');
  setTool(null);
  try {
    state.job = await post('/api/jobs', runBody());
  } catch (error) {
    showError(error.message);
    return;
  }
  $('progress').hidden = false;
  $('cancel-button').hidden = false;
  updateRunButton();
  showProgress(state.job);
  state.poll = setInterval(poll, 400);
}

async function poll() {
  if (!state.job) return;
  try {
    const status = await api(`/api/jobs/${state.job.id}`);
    state.job = status;
    showProgress(status);
    if (['done', 'failed', 'cancelled'].includes(status.state)) {
      stopPolling();
      if (status.state === 'done') await showResults(status);
      if (status.state === 'failed') showError(status.error);
      if (status.state === 'cancelled') setBusy('Cancelled');
    }
  } catch (error) {
    stopPolling();
    showError(error.message);
  }
}

function stopPolling() {
  if (state.poll) clearInterval(state.poll);
  state.poll = null;
  $('cancel-button').hidden = true;
  updateRunButton();
}

function showProgress(status) {
  const percent = Math.round((status.progress || 0) * 100);
  $('progress-fill').style.width = `${percent}%`;
  const words = { queued: 'Waiting', running: 'Running', done: 'Done', failed: 'Failed', cancelled: 'Cancelled' }[status.state];
  $('progress-text').textContent = `${words} · ${percent}% · ${status.elapsed.toFixed(0)} s`;
  setBusy(status.state === 'running' ? `Running sweep… ${percent}%` : '');
}

async function cancel() {
  if (state.job) await post(`/api/jobs/${state.job.id}/cancel`, {});
}

// ---------------------------------------------------------------- results

async function showResults(status) {
  state.result = status.result;
  state.stale = false;
  state.geometries.forEach((geometry) => geometry.dispose());
  state.geometries.clear();
  const stations = state.result.stations;
  const slider = $('station-slider');
  slider.max = stations.length - 1;
  const ticks = $('slider-ticks');
  ticks.replaceChildren(...stations.map((s) => {
    const tick = document.createElement('span');
    tick.textContent = pct(s.removed);
    return tick;
  }));
  $('sweep-note').hidden = !state.result.note;
  $('sweep-note').textContent = state.result.note ? `The sweep ${state.result.note}.` : '';
  $('results').hidden = false;
  $('results').classList.remove('is-stale');
  $('stale').hidden = true;
  await selectStation(Math.floor((stations.length - 1) / 2));
  prefetch();
}

async function prefetch() {
  const job = state.job;
  for (let i = 0; i < state.result.stations.length; i++) {
    if (state.job !== job) return;
    if (!state.geometries.has(i)) await stationGeometry(i);
  }
}

async function stationGeometry(index) {
  if (!state.geometries.has(index)) {
    const payload = await api(`/api/jobs/${state.job.id}/stations/${index}`);
    const geometry = geometryFrom(payload);
    colorize(geometry);
    state.geometries.set(index, geometry);
  }
  return state.geometries.get(index);
}

async function selectStation(index) {
  state.station = index;
  $('station-slider').value = index;
  const station = state.result.stations[index];
  $('station-removed').textContent = pct(station.removed);
  $('download-button').href = `/api/jobs/${state.job.id}/stations/${index}/stl`;
  renderTiles(station);
  renderCharts();
  renderTable();
  try {
    const geometry = await stationGeometry(index);
    if (state.station !== index) return;
    clear(layers.station, false);
    const useHeat = $('heat-toggle').checked && geometry.getAttribute('color');
    layers.station.add(new THREE.Mesh(geometry, useHeat ? heatMaterial : partMaterial));
  } catch (error) {
    showError(error.message);
  }
  updateView();
}

function renderTiles(station) {
  const m = station.metrics;
  const first = state.result.stations[0].metrics;
  const n = (value, digits, unit = '') => (value == null ? '—' : `${value.toFixed(digits)}${unit}`);
  const tiles = state.job.kind === 'load'
    ? [
        ['Mass', n(m.mass_g, 1, ' g'), `${n(m.mass_g - first.mass_g, 1, ' g')} vs original`],
        ['Stiffness kept', pct(m.stiffness), ''],
        ['Safety factor', n(m.safety_factor, 1, '×'), m.safety_factor != null && m.safety_factor < 3 ? 'print one and test it' : ''],
        ['Max deflection', n(m.deflection_mm, 3, ' mm'), ''],
        ['Peak stress', n(m.peak_stress_mpa, 1, ' MPa'), `strength ${state.result.strength} MPa`],
      ]
    : [
        ['Drag', pct(m.drag), 'of the original'],
        ['Drag force', n(m.force_n, 1, ' N'), `at ${$('speed-input').value} km/h`],
        ['Cd', n(m.cd, 3), 'on frontal area'],
        ['CdA', n(m.cda_m2, 4, ' m²'), ''],
      ];
  $('tiles').replaceChildren(...tiles.map(([label, value, delta]) => {
    const tile = document.createElement('div');
    tile.className = 'tile';
    const l = document.createElement('div'); l.className = 'tile-label'; l.textContent = label;
    const v = document.createElement('div'); v.className = 'tile-value'; v.textContent = value;
    const d = document.createElement('div'); d.className = 'tile-delta'; d.textContent = delta;
    tile.append(l, v, d);
    return tile;
  }));
}

function renderCharts() {
  const charts = $('charts');
  charts.replaceChildren();
  const stations = state.result.stations;
  const xs = stations.map((s) => s.removed);
  const series = (key) => stations.map((s) => s.metrics[key]);
  if (state.job.kind === 'load') {
    charts.append(lineChart('Stiffness kept', xs, series('stiffness'), { format: pct, max: 1 }));
    charts.append(lineChart('Safety factor', xs, series('safety_factor'), {
      format: (v) => `${v.toFixed(1)}×`, threshold: { value: 3, label: '3× — test below' },
    }));
  } else {
    charts.append(lineChart('Drag, share of the original', xs, series('drag'), { format: pct, max: 1 }));
    charts.append(lineChart(`Drag force at ${$('speed-input').value} km/h`, xs, series('force_n'), {
      format: (v) => `${v.toFixed(1)} N`,
    }));
  }
}

function lineChart(title, xs, ys, { format, max, threshold }) {
  const wrap = document.createElement('div');
  wrap.className = 'chart';
  const heading = document.createElement('p');
  heading.className = 'chart-title';
  heading.textContent = title;
  wrap.append(heading);
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  wrap.append(svg);
  const draw = () => {
    svg.replaceChildren();
    const width = svg.clientWidth || 320;
    const height = 140;
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const pad = { left: 44, right: 14, top: 8, bottom: 22 };
    const values = ys.map((y) => (y == null ? 0 : y));
    const top = niceCeil(Math.max(max ?? 0, ...values, threshold ? threshold.value * 1.15 : 0));
    const xMax = Math.max(...xs) || 1;
    const x = (v) => pad.left + (v / xMax) * (width - pad.left - pad.right);
    const y = (v) => height - pad.bottom - (v / top) * (height - pad.top - pad.bottom);
    const add = (tag, attrs, text) => {
      const node = document.createElementNS(ns, tag);
      for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
      if (text != null) node.textContent = text;
      svg.append(node);
      return node;
    };
    for (let i = 0; i <= 4; i++) {
      const value = (top * i) / 4;
      add('line', { class: i === 0 ? 'baseline' : 'gridline', x1: pad.left, x2: width - pad.right, y1: y(value), y2: y(value) });
      add('text', { class: 'tick', x: pad.left - 6, y: y(value) + 4, 'text-anchor': 'end' }, format(value));
    }
    for (const tick of [0, 0.25, 0.5, 0.75, 1].map((f) => f * xMax)) {
      add('text', { class: 'tick', x: x(tick), y: height - 6, 'text-anchor': 'middle' }, pct(tick));
    }
    if (threshold && threshold.value <= top) {
      add('line', { class: 'threshold', x1: pad.left, x2: width - pad.right, y1: y(threshold.value), y2: y(threshold.value) });
      add('text', { class: 'threshold-label', x: width - pad.right, y: y(threshold.value) - 4, 'text-anchor': 'end' }, threshold.label);
    }
    const path = values.map((v, i) => `${i ? 'L' : 'M'}${x(xs[i]).toFixed(1)},${y(v).toFixed(1)}`).join('');
    add('path', { class: 'area', d: `${path}L${x(xs[xs.length - 1])},${y(0)}L${x(xs[0])},${y(0)}Z` });
    add('path', { class: 'series', d: path });
    values.forEach((v, i) => {
      add('circle', { class: i === state.station ? 'current' : 'point', cx: x(xs[i]), cy: y(v), r: i === state.station ? 6 : 4 });
    });
    const cursor = add('line', { class: 'cursor', y1: pad.top, y2: height - pad.bottom, visibility: 'hidden' });
    const hit = add('rect', { class: 'hit', x: pad.left - 10, y: 0, width: width - pad.left - pad.right + 20, height });
    const nearest = (event) => {
      const rect = svg.getBoundingClientRect();
      const px = ((event.clientX - rect.left) / rect.width) * width;
      let best = 0;
      xs.forEach((value, i) => { if (Math.abs(x(value) - px) < Math.abs(x(xs[best]) - px)) best = i; });
      return best;
    };
    hit.addEventListener('pointermove', (event) => {
      const i = nearest(event);
      cursor.setAttribute('x1', x(xs[i])); cursor.setAttribute('x2', x(xs[i]));
      cursor.setAttribute('visibility', 'visible');
      showTooltip(event, ys[i] == null ? '—' : format(ys[i]), `${title} at ${pct(xs[i])} removed`);
    });
    hit.addEventListener('pointerleave', () => { cursor.setAttribute('visibility', 'hidden'); hideTooltip(); });
    hit.addEventListener('click', (event) => selectStation(nearest(event)));
  };
  requestAnimationFrame(draw);
  wrap.redraw = draw;
  return wrap;
}

function niceCeil(value) {
  if (value <= 0) return 1;
  const step = niceStep(value / 4);
  return Math.ceil(value / step) * step;
}

function renderTable() {
  const table = $('stations-table');
  const columns = [{ key: 'removed', label: 'removed', format: '{:.0%}' }, ...state.result.columns];
  const head = document.createElement('tr');
  for (const c of columns) {
    const th = document.createElement('th');
    th.textContent = c.label;
    head.append(th);
  }
  const rows = state.result.stations.map((station, index) => {
    const tr = document.createElement('tr');
    tr.className = index === state.station ? 'row current' : 'row';
    tr.tabIndex = 0;
    for (const c of columns) {
      const td = document.createElement('td');
      td.textContent = fmt(c.format, c.key === 'removed' ? station.removed : station.metrics[c.key]);
      tr.append(td);
    }
    tr.addEventListener('click', () => selectStation(index));
    tr.addEventListener('keydown', (event) => { if (event.key === 'Enter') selectStation(index); });
    return tr;
  });
  table.replaceChildren(head, ...rows);
}

// Python format specs, as the server sends them: "{:.1f}", "{:.0%}", "{:.1f}x".
function fmt(spec, value) {
  if (value == null) return '—';
  const match = /^\{:\.(\d+)([f%])\}(.*)$/.exec(spec);
  if (!match) return String(value);
  const digits = Number(match[1]);
  return (match[2] === '%' ? `${(value * 100).toFixed(digits)}%` : value.toFixed(digits)) + match[3];
}

const pct = (v) => `${Math.round(v * 100)}%`;

function showTooltip(event, value, label) {
  const tip = $('tooltip');
  tip.replaceChildren();
  const strong = document.createElement('strong');
  strong.textContent = value;
  const line = document.createElement('div');
  const key = document.createElement('span');
  key.className = 'key';
  line.append(key, document.createTextNode(label));
  tip.append(strong, line);
  tip.hidden = false;
  tip.style.left = `${event.clientX + 14}px`;
  tip.style.top = `${event.clientY - 10}px`;
}

function hideTooltip() { $('tooltip').hidden = true; }

// ---------------------------------------------------------------- view state

function updateView() {
  const showStation = Boolean(state.result) && !state.tool && !$('results').hidden;
  layers.original.visible = Boolean(state.part) && !showStation;
  layers.station.visible = showStation;
  layers.ghost.visible = showStation && $('ghost-toggle').checked;
  layers.markers.visible = state.kind === 'load';
  layers.flow.visible = state.kind === 'flow';
  drawLegend();
  updateHint();
}

function updateHint() {
  let hint = '';
  if (state.part && state.tool) {
    const what = { support: 'where it is held', load: 'where the load goes', keep: 'material to keep' }[state.tool];
    hint = `Click the part: ${what} — ${state.select === 'face' ? 'whole face' : 'around the point'}`;
  } else if (state.part && !state.result) {
    hint = 'Drag to orbit · right-drag to pan · scroll to zoom';
  }
  $('hud-hint').textContent = hint;
}

function markStale() {
  if (!state.result || state.stale) return;
  state.stale = true;
  $('results').classList.add('is-stale');
  $('stale').hidden = false;
}

function setBusy(text) { $('topbar-status').textContent = text; }

function showError(message) {
  $('error').hidden = !message;
  $('error').textContent = message;
}

function syncOutputs() {
  $('remove-out').textContent = `${$('remove-input').value}%`;
  for (const kind of ['load', 'flow']) {
    const input = $(`${kind}-resolution`);
    let text = `${input.value} voxels`;
    if (state.part) {
      const s = bounds().size;
      text += ` · ${fmtLength(Math.max(s.x, s.y, s.z) / Number(input.value))} mm`;
    }
    $(`${kind}-resolution-out`).textContent = text;
  }
}

// ---------------------------------------------------------------- wiring

$('example-select').addEventListener('change', (event) => openExample(event.target.value));
$('file-input').addEventListener('change', (event) => openFile(event.target.files[0]));
$('units-select').addEventListener('change', () => { if (state.file) openFile(state.file); });
document.querySelectorAll('[data-kind]').forEach((b) => b.addEventListener('click', () => setKind(b.dataset.kind)));
document.querySelectorAll('.tool').forEach((b) => b.addEventListener('click', () => setTool(b.dataset.tool)));
document.querySelectorAll('[data-select]').forEach((b) => b.addEventListener('click', () => setSelect(b.dataset.select)));
document.querySelectorAll('[data-flow]').forEach((b) => b.addEventListener('click', () => setFlow(b.dataset.flow.split(',').map(Number))));
$('force-direction').addEventListener('change', (event) => { $('custom-direction').hidden = event.target.value !== 'custom'; });
$('keep-box-toggle').addEventListener('change', (event) => { $('keep-box-fields').hidden = !event.target.checked; drawFlow(); markStale(); });
document.querySelectorAll('#keep-box-fields input').forEach((i) => i.addEventListener('change', () => { drawFlow(); markStale(); }));
document.querySelectorAll('#flow-panel select, #flow-panel input, #material-select').forEach((i) => i.addEventListener('change', markStale));
$('remove-input').addEventListener('input', () => { syncOutputs(); markStale(); });
$('stations-input').addEventListener('change', markStale);
$('flow-resolution').addEventListener('input', syncOutputs);
let previewTimer = null;
$('load-resolution').addEventListener('input', () => {
  syncOutputs();
  markStale();
  clearTimeout(previewTimer);
  previewTimer = setTimeout(refreshPreviews, 350);
});
$('run-button').addEventListener('click', run);
$('cancel-button').addEventListener('click', cancel);
$('station-slider').addEventListener('input', (event) => selectStation(Number(event.target.value)));
$('heat-toggle').addEventListener('change', () => { if (state.result) selectStation(state.station); });
$('ghost-toggle').addEventListener('change', updateView);
$('reset-view').addEventListener('click', () => { if (state.part) frame(); });
window.addEventListener('resize', () => document.querySelectorAll('.chart').forEach((c) => c.redraw?.()));

viewport.addEventListener('dragover', (event) => { event.preventDefault(); $('drop-veil').hidden = false; });
viewport.addEventListener('dragleave', () => { $('drop-veil').hidden = true; });
viewport.addEventListener('drop', (event) => {
  event.preventDefault();
  $('drop-veil').hidden = true;
  openFile(event.dataTransfer.files[0]);
});

setSelect('face');
setTool(null);
syncOutputs();
loadInfo().catch((error) => showError(`Could not reach the whittle server: ${error.message}`));
