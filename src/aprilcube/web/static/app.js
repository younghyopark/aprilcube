(function () {
  "use strict";

  const MAX_GRID_SIZE = 12;
  const MARKER_DATA = window.APRILCUBE_MARKERS || {};
  const DICTIONARIES = Object.entries(MARKER_DATA)
    .map(([name, data]) => ({
      name,
      marker_pixels: data.marker_pixels,
      max_ids: data.max_ids,
      data: data.data,
      bit_count: data.bit_count,
    }))
    .sort((a, b) => a.max_ids - b.max_ids || a.marker_pixels - b.marker_pixels || a.name.localeCompare(b.name));

  const FACE_DEFS = [
    { name: "+X", axis: 0, sign: 1, delta: [1, 0, 0], normal: [1, 0, 0] },
    { name: "-X", axis: 0, sign: -1, delta: [-1, 0, 0], normal: [-1, 0, 0] },
    { name: "+Y", axis: 1, sign: 1, delta: [0, 1, 0], normal: [0, 1, 0] },
    { name: "-Y", axis: 1, sign: -1, delta: [0, -1, 0], normal: [0, -1, 0] },
    { name: "+Z", axis: 2, sign: 1, delta: [0, 0, 1], normal: [0, 0, 1] },
    { name: "-Z", axis: 2, sign: -1, delta: [0, 0, -1], normal: [0, 0, -1] },
  ];

  const els = {
    editorViewport: document.querySelector("#editor-viewport"),
    previewViewport: document.querySelector("#preview-viewport"),
    name: document.querySelector("#target-name"),
    dimX: document.querySelector("#dim-x"),
    dimY: document.querySelector("#dim-y"),
    dimZ: document.querySelector("#dim-z"),
    layerButtons: document.querySelector("#layer-buttons"),
    showAllToggle: document.querySelector("#show-all-toggle"),
    editorLayerStatus: document.querySelector("#editor-layer-status"),
    clear: document.querySelector("#clear-button"),
    fillLayer: document.querySelector("#fill-layer-button"),
    fill: document.querySelector("#fill-button"),
    dictionary: document.querySelector("#dictionary-select"),
    voxelSize: document.querySelector("#voxel-size"),
    tagSize: document.querySelector("#tag-size"),
    marginCells: document.querySelector("#margin-cells"),
    borderCells: document.querySelector("#border-cells"),
    extruder: document.querySelector("#extruder"),
    invert: document.querySelector("#invert"),
    voxelCount: document.querySelector("#voxel-count"),
    markerCount: document.querySelector("#marker-count"),
    dictionaryStatus: document.querySelector("#dictionary-status"),
    exportButton: document.querySelector("#export-button"),
    status: document.querySelector("#status-line"),
    hoverLabel: document.querySelector("#hover-label"),
    previewStatus: document.querySelector("#preview-status"),
    commandModal: document.querySelector("#command-modal"),
    commandText: document.querySelector("#command-text"),
    modalSubtitle: document.querySelector("#command-modal-subtitle"),
    output3mf: document.querySelector("#output-3mf"),
    outputXml: document.querySelector("#output-xml"),
    outputObj: document.querySelector("#output-obj"),
    copyCommand: document.querySelector("#copy-command"),
    modalClose: document.querySelector("#modal-close"),
    modalDone: document.querySelector("#modal-done"),
  };

  const state = {
    dims: [3, 1, 3],
    activeLayer: 0,
    showAllLayers: true,
    voxels: new Set(["1,0,0", "1,0,1", "0,0,2", "1,0,2", "2,0,2"]),
    dictionaryChoice: null,
    markerCache: new Map(),
    previewTexture: null,
    previewMesh: null,
    previewEdges: null,
    lastHover: null,
    pointerDown: null,
  };

  const editor = createViewport(els.editorViewport, 42);
  const preview = createViewport(els.previewViewport, 38);

  const editorRoot = new THREE.Group();
  const editLayerGroup = new THREE.Group();
  const occupiedGroup = new THREE.Group();
  const guideGroup = new THREE.Group();
  editorRoot.add(guideGroup, occupiedGroup, editLayerGroup);
  editor.scene.add(editorRoot);

  const previewRoot = new THREE.Group();
  preview.scene.add(previewRoot);

  const cellGeometry = new THREE.BoxGeometry(0.92, 0.92, 0.92);
  const voxelEdgeGeometry = new THREE.EdgesGeometry(cellGeometry);
  const editGeometry = new THREE.BoxGeometry(0.98, 0.98, 0.98);
  const highlightGeometry = new THREE.BoxGeometry(1.04, 1.04, 1.04);
  const occupiedMaterial = new THREE.MeshBasicMaterial({
    color: 0xf06a2f,
  });
  const ghostMaterial = new THREE.MeshBasicMaterial({
    color: 0xd9a84f,
    transparent: true,
    opacity: 0.34,
    depthWrite: false,
  });
  const editMaterial = new THREE.MeshBasicMaterial({
    color: 0x0e7c7b,
    transparent: true,
    opacity: 0.13,
    wireframe: true,
  });
  const highlightMaterial = new THREE.MeshBasicMaterial({
    color: 0x0e7c7b,
    transparent: true,
    opacity: 0.28,
    wireframe: true,
  });
  const planeMaterial = new THREE.MeshBasicMaterial({
    color: 0x9eb4b1,
    transparent: true,
    opacity: 0.08,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const previewMaterial = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.62,
    metalness: 0.0,
    side: THREE.DoubleSide,
  });
  const edgeMaterial = new THREE.LineBasicMaterial({
    color: 0x415252,
    transparent: true,
    opacity: 0.24,
  });
  const activeVoxelEdgeMaterial = new THREE.LineBasicMaterial({
    color: 0x7a321e,
    transparent: true,
    opacity: 0.72,
  });
  const ghostVoxelEdgeMaterial = new THREE.LineBasicMaterial({
    color: 0x7f8d87,
    transparent: true,
    opacity: 0.32,
  });

  const highlight = new THREE.Mesh(highlightGeometry, highlightMaterial);
  highlight.visible = false;
  editor.scene.add(highlight);

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();

  function createViewport(container, fov) {
    const scene = new THREE.Scene();
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputEncoding = THREE.sRGBEncoding;
    container.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(fov, 1, 0.1, 1000);
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const hemi = new THREE.HemisphereLight(0xffffff, 0x8fa8a1, 2.4);
    scene.add(hemi);
    const sun = new THREE.DirectionalLight(0xffffff, 2.6);
    sun.position.set(5, 8, 4);
    scene.add(sun);

    return { container, scene, renderer, camera, controls };
  }

  function key(x, y, z) {
    return `${x},${y},${z}`;
  }

  function parseKey(value) {
    return value.split(",").map((part) => Number.parseInt(part, 10));
  }

  function positionFor(x, y, z) {
    const [dx, dy, dz] = state.dims;
    return new THREE.Vector3(
      x - (dx - 1) / 2,
      z - (dz - 1) / 2,
      y - (dy - 1) / 2,
    );
  }

  function boundaryToThree(point) {
    const [dx, dy, dz] = state.dims;
    return [
      point[0] - dx / 2,
      point[2] - dz / 2,
      point[1] - dy / 2,
    ];
  }

  function normalToThree(normal) {
    return [normal[0], normal[2], normal[1]];
  }

  function clearGroup(group) {
    while (group.children.length) {
      const child = group.children.pop();
      if (
        child.geometry
        && child.geometry !== cellGeometry
        && child.geometry !== voxelEdgeGeometry
        && child.geometry !== editGeometry
      ) {
        child.geometry.dispose();
      }
    }
  }

  function rebuildEditor(resetCamera = false) {
    clearGroup(editLayerGroup);
    clearGroup(occupiedGroup);
    clearGroup(guideGroup);

    const [dx, dy] = state.dims;
    const activeZ = state.activeLayer;
    const showAll = state.showAllLayers;

    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(Math.max(dx, 1), Math.max(dy, 1)),
      planeMaterial,
    );
    plane.rotation.x = -Math.PI / 2;
    plane.position.copy(positionFor((dx - 1) / 2, (dy - 1) / 2, activeZ));
    guideGroup.add(plane);

    for (const raw of state.voxels) {
      const [x, y, z] = parseKey(raw);
      if (!showAll && z !== activeZ) {
        continue;
      }
      const mat = z === activeZ ? occupiedMaterial : ghostMaterial;
      const mesh = new THREE.Mesh(cellGeometry, mat);
      mesh.position.copy(positionFor(x, y, z));
      occupiedGroup.add(mesh);

      const edgeMat = z === activeZ ? activeVoxelEdgeMaterial : ghostVoxelEdgeMaterial;
      const edges = new THREE.LineSegments(voxelEdgeGeometry, edgeMat);
      edges.position.copy(mesh.position);
      occupiedGroup.add(edges);
    }

    for (let x = 0; x < dx; x += 1) {
      for (let y = 0; y < dy; y += 1) {
        const mesh = new THREE.Mesh(editGeometry, editMaterial);
        mesh.position.copy(positionFor(x, y, activeZ));
        mesh.userData.voxel = [x, y, activeZ];
        editLayerGroup.add(mesh);
      }
    }

    highlight.visible = false;
    state.lastHover = null;
    if (resetCamera) {
      fitCamera(editor, 1.22);
    }
  }

  function rebuildPreview(resetCamera = false) {
    if (state.previewMesh) {
      previewRoot.remove(state.previewMesh);
      state.previewMesh.geometry.dispose();
      state.previewMesh = null;
    }
    if (state.previewEdges) {
      previewRoot.remove(state.previewEdges);
      state.previewEdges.geometry.dispose();
      state.previewEdges = null;
    }
    if (state.previewTexture) {
      state.previewTexture.dispose();
      state.previewTexture = null;
    }

    const exposed = exposedFaces();
    if (!exposed.length || !state.dictionaryChoice || !state.dictionaryChoice.ok) {
      els.previewStatus.textContent = exposed.length ? "Adjust settings" : "No voxels";
      return;
    }

    const atlas = buildMarkerAtlas(state.dictionaryChoice.info, exposed.length);
    state.previewTexture = new THREE.CanvasTexture(atlas.canvas);
    state.previewTexture.encoding = THREE.sRGBEncoding;
    state.previewTexture.anisotropy = 4;
    previewMaterial.map = state.previewTexture;
    previewMaterial.needsUpdate = true;

    const geometry = buildTexturedGeometry(exposed, atlas);
    state.previewMesh = new THREE.Mesh(geometry, previewMaterial);
    previewRoot.add(state.previewMesh);

    state.previewEdges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), edgeMaterial);
    previewRoot.add(state.previewEdges);

    els.previewStatus.textContent = `${exposed.length} textured faces`;
    if (resetCamera) {
      fitCamera(preview, 1.12);
    }
  }

  function fitCamera(view, padding) {
    const [dx, dy, dz] = state.dims;
    const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2 + 0.75;
    const fov = THREE.MathUtils.degToRad(view.camera.fov);
    const aspect = Math.max(view.camera.aspect || 1, 0.1);
    const verticalDistance = radius / Math.sin(fov / 2);
    const horizontalFov = 2 * Math.atan(Math.tan(fov / 2) * aspect);
    const horizontalDistance = radius / Math.sin(horizontalFov / 2);
    const distance = Math.max(verticalDistance, horizontalDistance) * padding;
    const direction = new THREE.Vector3(1.25, 0.85, 1.55).normalize();

    view.controls.target.set(0, 0, 0);
    view.camera.position.copy(direction.multiplyScalar(distance));
    view.camera.near = Math.max(0.01, distance / 100);
    view.camera.far = distance * 100;
    view.camera.updateProjectionMatrix();
    view.controls.update();
  }

  function updateDimensions() {
    const dims = [
      clampInteger(els.dimX.value, 1, MAX_GRID_SIZE),
      clampInteger(els.dimY.value, 1, MAX_GRID_SIZE),
      clampInteger(els.dimZ.value, 1, MAX_GRID_SIZE),
    ];
    state.dims = dims;
    els.dimX.value = dims[0];
    els.dimY.value = dims[1];
    els.dimZ.value = dims[2];

    state.voxels = new Set([...state.voxels].filter((raw) => {
      const [x, y, z] = parseKey(raw);
      return x < dims[0] && y < dims[1] && z < dims[2];
    }));

    state.activeLayer = Math.min(state.activeLayer, dims[2] - 1);
    buildLayerRail();
    refreshAll(true);
  }

  function refreshAll(resetCamera = false) {
    updateStats();
    rebuildEditor(resetCamera);
    rebuildPreview(resetCamera);
  }

  function updateLayerLabel() {
    const prefix = state.showAllLayers ? "All layers" : "Single layer";
    els.editorLayerStatus.textContent = `${prefix} / editing Z ${state.activeLayer}`;
    els.showAllToggle.classList.toggle("active", state.showAllLayers);
    els.showAllToggle.setAttribute("aria-pressed", String(state.showAllLayers));
  }

  function buildLayerRail() {
    els.layerButtons.innerHTML = "";
    for (let z = state.dims[2] - 1; z >= 0; z -= 1) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "layer-button";
      button.dataset.layer = String(z);
      button.textContent = String(z);
      button.setAttribute("aria-label", `Edit Z layer ${z}`);
      button.addEventListener("click", () => {
        state.activeLayer = z;
        updateLayerLabel();
        updateLayerRailState();
        rebuildEditor(false);
      });
      els.layerButtons.appendChild(button);
    }
    updateLayerLabel();
    updateLayerRailState();
  }

  function updateLayerRailState() {
    const occupiedLayers = new Set([...state.voxels].map((raw) => parseKey(raw)[2]));
    for (const button of els.layerButtons.querySelectorAll(".layer-button")) {
      const z = Number.parseInt(button.dataset.layer, 10);
      button.classList.toggle("active", z === state.activeLayer);
      button.classList.toggle("has-voxels", occupiedLayers.has(z));
      button.setAttribute("aria-pressed", String(z === state.activeLayer));
    }
  }

  function clampInteger(value, min, max) {
    const parsed = Number.parseInt(value, 10);
    if (Number.isNaN(parsed)) {
      return min;
    }
    return Math.max(min, Math.min(max, parsed));
  }

  function numberValue(input, fallback) {
    const parsed = Number.parseFloat(input.value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function integerValue(input, fallback) {
    const parsed = Number.parseInt(input.value, 10);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function exposedFaces() {
    const faces = [];
    const sortedVoxels = [...state.voxels].map(parseKey).sort((a, b) => (
      a[0] - b[0] || a[1] - b[1] || a[2] - b[2]
    ));
    for (const voxel of sortedVoxels) {
      for (const face of FACE_DEFS) {
        const neighbor = [
          voxel[0] + face.delta[0],
          voxel[1] + face.delta[1],
          voxel[2] + face.delta[2],
        ];
        if (!state.voxels.has(key(...neighbor))) {
          faces.push({ voxel, face });
        }
      }
    }
    return faces;
  }

  function compatibleDictionary(info) {
    const voxelSize = numberValue(els.voxelSize, 24);
    const tagSize = numberValue(els.tagSize, 18);
    const border = integerValue(els.borderCells, 1);
    const cellSize = tagSize / info.marker_pixels;
    if (cellSize <= 0) {
      return false;
    }
    const faceCells = voxelSize / cellSize;
    const rounded = Math.round(faceCells);
    return Math.abs(faceCells - rounded) < 1e-6
      && rounded >= info.marker_pixels + 2 * border;
  }

  function resolveDictionary(needed) {
    const selected = els.dictionary.value;
    if (selected !== "auto") {
      const info = DICTIONARIES.find((item) => item.name === selected);
      if (!info) {
        return { name: selected, ok: false, message: "Unknown" };
      }
      if (info.max_ids < needed) {
        return { name: selected, ok: false, message: `${info.max_ids} ids` };
      }
      if (!compatibleDictionary(info)) {
        return { name: selected, ok: false, message: "Size mismatch" };
      }
      return { name: selected, ok: true, message: selected, info };
    }

    const info = DICTIONARIES.find((item) => item.max_ids >= needed && compatibleDictionary(item));
    if (!info) {
      return { name: "auto", ok: false, message: "No fit" };
    }
    return { name: info.name, ok: true, message: info.name, info };
  }

  function updateStats() {
    const markerCount = exposedFaces().length;
    const dictionary = resolveDictionary(markerCount);
    state.dictionaryChoice = dictionary;

    els.voxelCount.textContent = String(state.voxels.size);
    els.markerCount.textContent = String(markerCount);
    els.dictionaryStatus.textContent = dictionary.message;

    const canExport = state.voxels.size > 0 && dictionary.ok;
    els.exportButton.disabled = !canExport;
    if (state.voxels.size === 0) {
      setStatus("Add at least one voxel.", true);
    } else if (!dictionary.ok) {
      setStatus("Adjust dictionary or marker sizing before downloading YAML.", true);
    } else {
      setStatus(`Download YAML, then run: aprilcube generate ${sanitizeName(els.name.value)}.yaml`, false);
    }
    updateLayerRailState();
  }

  function setStatus(message, isError = false) {
    els.status.textContent = message;
    els.status.classList.toggle("error", isError);
  }

  function buildMarkerAtlas(dictionary, count) {
    const tilePx = count > 500 ? 48 : count > 100 ? 64 : 96;
    const cols = Math.ceil(Math.sqrt(count));
    const rows = Math.ceil(count / cols);
    const canvas = document.createElement("canvas");
    canvas.width = cols * tilePx;
    canvas.height = rows * tilePx;
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const regions = [];
    for (let id = 0; id < count; id += 1) {
      const col = id % cols;
      const row = Math.floor(id / cols);
      const x = col * tilePx;
      const y = row * tilePx;
      drawMarkerTile(ctx, dictionary, id, x, y, tilePx);
      regions.push({ x, y, w: tilePx, h: tilePx });
    }
    return { canvas, regions };
  }

  function drawMarkerTile(ctx, dictionary, id, x, y, tilePx) {
    const markerPixels = dictionary.marker_pixels;
    const voxelSize = numberValue(els.voxelSize, 24);
    const tagSize = numberValue(els.tagSize, 18);
    const borderCells = integerValue(els.borderCells, 1);
    const cellSize = tagSize / markerPixels;
    const faceCells = Math.max(markerPixels + 2 * borderCells, Math.round(voxelSize / cellSize));
    const cellPx = tilePx / faceCells;
    const offset = Math.floor((faceCells - markerPixels) / 2);
    const invert = els.invert.checked;

    ctx.fillStyle = invert ? "#050505" : "#ffffff";
    ctx.fillRect(x, y, tilePx, tilePx);
    ctx.strokeStyle = "#d6dedb";
    ctx.lineWidth = Math.max(1, tilePx / 96);
    ctx.strokeRect(x + 0.5, y + 0.5, tilePx - 1, tilePx - 1);

    for (let row = 0; row < markerPixels; row += 1) {
      for (let col = 0; col < markerPixels; col += 1) {
        const isBlack = markerBit(dictionary, id, row * markerPixels + col);
        const black = invert ? !isBlack : isBlack;
        ctx.fillStyle = black ? "#050505" : "#ffffff";
        ctx.fillRect(
          x + (offset + col) * cellPx,
          y + (offset + row) * cellPx,
          Math.ceil(cellPx),
          Math.ceil(cellPx),
        );
      }
    }
  }

  function markerBit(dictionary, id, bitIndex) {
    let bytes = state.markerCache.get(dictionary.name);
    if (!bytes) {
      const binary = atob(dictionary.data);
      bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
      }
      state.markerCache.set(dictionary.name, bytes);
    }
    const markerBits = dictionary.marker_pixels * dictionary.marker_pixels;
    const globalBit = id * markerBits + bitIndex;
    const value = bytes[globalBit >> 3] || 0;
    return (value & (1 << (7 - (globalBit & 7)))) !== 0;
  }

  function buildTexturedGeometry(faces, atlas) {
    const positions = [];
    const normals = [];
    const uvs = [];
    const indices = [];

    for (let idx = 0; idx < faces.length; idx += 1) {
      const { voxel, face } = faces[idx];
      const corners = faceCorners(voxel, face).map(boundaryToThree);
      const normal = normalToThree(face.normal);
      const region = atlas.regions[idx];
      const u0 = region.x / atlas.canvas.width;
      const u1 = (region.x + region.w) / atlas.canvas.width;
      const v0 = 1 - (region.y + region.h) / atlas.canvas.height;
      const v1 = 1 - region.y / atlas.canvas.height;
      const base = positions.length / 3;

      for (const corner of corners) {
        positions.push(...corner);
        normals.push(...normal);
      }
      uvs.push(u0, v1, u1, v1, u1, v0, u0, v0);
      indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
    geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(indices);
    geometry.computeBoundingSphere();
    return geometry;
  }

  function faceCorners(voxel, face) {
    const [x, y, z] = voxel;
    const x0 = x;
    const x1 = x + 1;
    const y0 = y;
    const y1 = y + 1;
    const z0 = z;
    const z1 = z + 1;
    switch (face.name) {
      case "+X":
        return [[x1, y1, z1], [x1, y0, z1], [x1, y0, z0], [x1, y1, z0]];
      case "-X":
        return [[x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [x0, y0, z0]];
      case "+Y":
        return [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]];
      case "-Y":
        return [[x1, y0, z1], [x0, y0, z1], [x0, y0, z0], [x1, y0, z0]];
      case "+Z":
        return [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
      case "-Z":
        return [[x0, y1, z0], [x1, y1, z0], [x1, y0, z0], [x0, y0, z0]];
      default:
        return [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]];
    }
  }

  function updatePointer(event) {
    const rect = editor.renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  function pickVoxel(event) {
    updatePointer(event);
    raycaster.setFromCamera(pointer, editor.camera);
    const hits = raycaster.intersectObjects(editLayerGroup.children, false);
    return hits.length ? hits[0].object.userData.voxel : null;
  }

  function onPointerMove(event) {
    const voxel = pickVoxel(event);
    if (!voxel) {
      highlight.visible = false;
      state.lastHover = null;
      els.hoverLabel.textContent = "No voxel selected";
      return;
    }
    const [x, y, z] = voxel;
    highlight.position.copy(positionFor(x, y, z));
    highlight.visible = true;
    state.lastHover = voxel;
    els.hoverLabel.textContent = `Voxel [${x}, ${y}, ${z}]`;
  }

  function onPointerDown(event) {
    if (event.button !== 0) {
      return;
    }
    const voxel = pickVoxel(event);
    state.pointerDown = voxel
      ? { x: event.clientX, y: event.clientY, voxel }
      : null;
  }

  function onPointerUp(event) {
    if (event.button !== 0 || !state.pointerDown) {
      state.pointerDown = null;
      return;
    }
    const movement = Math.hypot(
      event.clientX - state.pointerDown.x,
      event.clientY - state.pointerDown.y,
    );
    if (movement > 5) {
      state.pointerDown = null;
      return;
    }
    const voxel = state.pointerDown.voxel;
    const raw = key(...voxel);
    if (state.voxels.has(raw)) {
      state.voxels.delete(raw);
    } else {
      state.voxels.add(raw);
    }
    state.pointerDown = null;
    refreshAll(false);
  }

  function fillLayer() {
    const [dx, dy] = state.dims;
    for (let x = 0; x < dx; x += 1) {
      for (let y = 0; y < dy; y += 1) {
        state.voxels.add(key(x, y, state.activeLayer));
      }
    }
    refreshAll(false);
  }

  function fillAll() {
    const [dx, dy, dz] = state.dims;
    for (let x = 0; x < dx; x += 1) {
      for (let y = 0; y < dy; y += 1) {
        for (let z = 0; z < dz; z += 1) {
          state.voxels.add(key(x, y, z));
        }
      }
    }
    refreshAll(false);
  }

  function clearShape() {
    state.voxels.clear();
    refreshAll(false);
  }

  function yamlSpec() {
    const name = sanitizeName(els.name.value);
    const faces = exposedFaces();
    const lines = [
      `output: models/${name}`,
      "shape:",
      "  type: voxel_grid",
      `  voxel_size_mm: ${numberValue(els.voxelSize, 24)}`,
      "  voxels:",
    ];
    const voxels = [...state.voxels].map(parseKey).sort((a, b) => (
      a[0] - b[0] || a[1] - b[1] || a[2] - b[2]
    ));
    for (const voxel of voxels) {
      lines.push(`    - [${voxel[0]}, ${voxel[1]}, ${voxel[2]}]`);
    }
    lines.push(
      `dictionary: ${state.dictionaryChoice.name}`,
      "markers:",
      `  ids: 0-${Math.max(0, faces.length - 1)}`,
      "size:",
      `  tag_size_mm: ${numberValue(els.tagSize, 18)}`,
      "layout:",
      `  margin_cells: ${integerValue(els.marginCells, 1)}`,
      `  border_cells: ${integerValue(els.borderCells, 1)}`,
      "material:",
      `  extruder: ${integerValue(els.extruder, 1)}`,
      `  invert: ${els.invert.checked ? "true" : "false"}`,
      "",
    );
    return lines.join("\n");
  }

  function downloadYaml() {
    if (!state.dictionaryChoice || !state.dictionaryChoice.ok || !state.voxels.size) {
      updateStats();
      return;
    }
    const name = sanitizeName(els.name.value);
    const blob = new Blob([yamlSpec()], { type: "application/x-yaml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${name}.yaml`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    setStatus(`Downloaded ${name}.yaml. Run: aprilcube generate ${name}.yaml`, false);
    showCommandModal(name);
  }

  function showCommandModal(name) {
    const yamlFile = `${name}.yaml`;
    const outputDir = `models/${name}`;
    const command = [
      "python -m pip install aprilcube",
      "cd ~/Downloads",
      `aprilcube generate ${yamlFile}`,
    ].join("\n");

    els.modalSubtitle.textContent = `Assuming ${yamlFile} was saved to Downloads.`;
    els.commandText.textContent = command;
    els.output3mf.textContent = `${outputDir}/cube.3mf`;
    els.outputXml.textContent = `${outputDir}/mujoco/cube.xml`;
    els.outputObj.textContent = `${outputDir}/mujoco/cube.obj`;
    els.copyCommand.textContent = "Copy Command";
    els.commandModal.hidden = false;
    els.modalDone.focus();
  }

  function closeCommandModal() {
    els.commandModal.hidden = true;
  }

  async function copyCommand() {
    const text = els.commandText.textContent;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.setAttribute("readonly", "");
        textarea.style.position = "fixed";
        textarea.style.left = "-9999px";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        textarea.remove();
      }
      els.copyCommand.textContent = "Copied";
    } catch (error) {
      els.copyCommand.textContent = "Copy Failed";
    }
  }

  function sanitizeName(value) {
    const cleaned = String(value || "aprilcube_target")
      .trim()
      .replace(/[^A-Za-z0-9._-]+/g, "_")
      .replace(/^[._-]+|[._-]+$/g, "");
    return cleaned.slice(0, 80) || "aprilcube_target";
  }

  function resizeViewport(view) {
    const width = view.container.clientWidth;
    const height = view.container.clientHeight;
    if (width <= 0 || height <= 0) {
      return;
    }
    view.renderer.setSize(width, height);
    view.camera.aspect = width / Math.max(height, 1);
    view.camera.updateProjectionMatrix();
  }

  function resizeAll(refit = false) {
    resizeViewport(editor);
    resizeViewport(preview);
    if (refit) {
      fitCamera(editor, 1.22);
      fitCamera(preview, 1.12);
    }
  }

  function animate() {
    editor.controls.update();
    preview.controls.update();
    editor.renderer.render(editor.scene, editor.camera);
    preview.renderer.render(preview.scene, preview.camera);
    requestAnimationFrame(animate);
  }

  function populateDictionaries() {
    for (const info of DICTIONARIES) {
      const option = document.createElement("option");
      option.value = info.name;
      option.textContent = `${info.name} (${info.max_ids})`;
      els.dictionary.appendChild(option);
    }
  }

  for (const input of [els.dimX, els.dimY, els.dimZ]) {
    input.addEventListener("change", updateDimensions);
  }

  els.showAllToggle.addEventListener("click", () => {
    state.showAllLayers = !state.showAllLayers;
    updateLayerLabel();
    rebuildEditor(false);
  });
  els.clear.addEventListener("click", clearShape);
  els.fillLayer.addEventListener("click", fillLayer);
  els.fill.addEventListener("click", fillAll);
  els.dictionary.addEventListener("change", () => refreshAll(false));
  els.voxelSize.addEventListener("change", () => refreshAll(false));
  els.tagSize.addEventListener("change", () => refreshAll(false));
  els.marginCells.addEventListener("change", () => refreshAll(false));
  els.borderCells.addEventListener("change", () => refreshAll(false));
  els.extruder.addEventListener("change", updateStats);
  els.invert.addEventListener("change", () => refreshAll(false));
  els.exportButton.addEventListener("click", downloadYaml);
  els.modalClose.addEventListener("click", closeCommandModal);
  els.modalDone.addEventListener("click", closeCommandModal);
  els.copyCommand.addEventListener("click", copyCommand);
  els.commandModal.addEventListener("click", (event) => {
    if (event.target === els.commandModal) {
      closeCommandModal();
    }
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !els.commandModal.hidden) {
      closeCommandModal();
    }
  });

  editor.renderer.domElement.addEventListener("pointermove", onPointerMove);
  editor.renderer.domElement.addEventListener("pointerdown", onPointerDown);
  editor.renderer.domElement.addEventListener("pointerup", onPointerUp);
  editor.renderer.domElement.addEventListener("pointerleave", () => {
    state.pointerDown = null;
  });
  window.addEventListener("resize", () => resizeAll(true));

  new ResizeObserver(() => resizeAll(true)).observe(els.editorViewport);
  new ResizeObserver(() => resizeAll(true)).observe(els.previewViewport);

  populateDictionaries();
  resizeAll(true);
  updateDimensions();
  requestAnimationFrame(() => {
    resizeAll(true);
    refreshAll(true);
  });
  animate();
}());
