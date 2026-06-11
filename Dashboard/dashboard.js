// ── State ──────────────────────────────────────────────
const STATE = {
  status: 'IDLE',
  bearing: null,
  fireLocation: null,
  dronePos: null,
  photos: { tower: [], drone: [], mission: [] },
  expanded: false,
  activeTab: 'tower',
  fireReported: false
};
let droneTimer = null, lbPhotos = [], lbIdx = 0, missionStart = null, timerInterval = null;

// ── Init ──────────────────────────────────────────────
window.onload = () => {
  connectSSE();
  pollStatus();
  pollPhotos();
  setInterval(pollStatus, 2000);
  setInterval(pollPhotos, 5000);
};

// ── API ───────────────────────────────────────────────
async function pollStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    applyState(d);
  } catch (e) {}
}

async function pollPhotos() {
  try {
    const r = await fetch('/api/photos');
    const d = await r.json();
    STATE.photos = d;
    renderGallery();
  } catch (e) {}
}

// ── SSE ───────────────────────────────────────────────
function connectSSE() {
  const ev = new EventSource('/api/events');
  ev.onmessage = e => {
    try {
      const d = JSON.parse(e.data);
      if (d.type === 'log') {
        addConsole(d.message);
        // Also detect state from raw log lines in browser
        detectStateFromLog(d.message);
      }
      if (d.type === 'new_photo') pollPhotos();
      if (d.type === 'init') pollStatus();
    } catch (err) {}
  };
  ev.onerror = () => setTimeout(connectSSE, 3000);
}

// Detect state transitions from raw log text
function detectStateFromLog(msg) {
  // Alert detection
  if (/ALARM|SMOKE ALERT|🚨|SMOKE DETECTED/i.test(msg)) {
    if (STATE.status === 'IDLE' || STATE.status === 'MONITORING') {
      STATE.status = 'ALERT';
      updateStatusBar();
      triggerAlert();
    }
  }

  if (/DISPATCHING DRONE|PHASE A.*BEARING|Taking off|Takeoff successful/i.test(msg)) {
    if (STATE.status !== 'FLYING' && STATE.status !== 'MISSION_COMPLETE') {
      STATE.status = 'FLYING';
      updateStatusBar();
      startFlying();
    }
  }
  
  if (/Phase C.*Step|PPO.*training|PHASE C.*PPO/i.test(msg)) {
    if (STATE.status !== 'FLYING' && STATE.status !== 'MISSION_COMPLETE') {
      STATE.status = 'FLYING';
      updateStatusBar();
      startFlying();
    }
  }
  
  if (/FIRE CONFIRMED|fire_confirmed.*true|Mission complete|Landed safely|✅ MISSION COMPLETE|🏁 Mission complete/i.test(msg)) {
    if (STATE.status !== 'MISSION_COMPLETE') {
      STATE.status = 'MISSION_COMPLETE';
      updateStatusBar();
      completeMission();
    }
  }

  // Parse fire coords from log
  const fireMatch = msg.match(/Fire.*?\(([\-0-9.]+),\s*([\-0-9.]+)\)/i);
  if (fireMatch) {
    STATE.fireLocation = { x: parseFloat(fireMatch[1]), y: parseFloat(fireMatch[2]) };
    if (STATE.status === 'MISSION_COMPLETE') {
      placeFireMarker(STATE.fireLocation.x, STATE.fireLocation.y);
    }
  }
  
  // Parse bearing
  const bMatch = msg.match(/[Bb]earing[:\s]+([0-9.]+)/i);
  if (bMatch && !STATE.bearing) {
    STATE.bearing = parseFloat(bMatch[1]);
    const coords = document.getElementById('sc-coords');
    if (coords) coords.textContent = `Bear: ${STATE.bearing.toFixed(1)}°`;
  }
}

// ── State machine ─────────────────────────────────────
function applyState(d) {
  const prev = STATE.status;
  STATE.status = (d.state || 'IDLE').toUpperCase();
  if (d.bearing) STATE.bearing = d.bearing;
  if (d.fire_location) STATE.fireLocation = d.fire_location;
  if (d.drone_position) STATE.dronePos = d.drone_position;

  updateStatusBar();

  if (STATE.status === 'ALERT' && prev !== 'ALERT') triggerAlert();
  if (STATE.status === 'FLYING' && prev !== 'FLYING') startFlying();
  if (STATE.status === 'MISSION_COMPLETE' && prev !== 'MISSION_COMPLETE') completeMission();
  if ((STATE.status === 'MONITORING' || STATE.status === 'IDLE') && prev === 'IDLE') goMonitoring();
}

function updateStatusBar() {
  const dot = document.getElementById('status-dot');
  const txt = document.getElementById('status-text');
  if (!dot || !txt) return;

  dot.className = 'status-dot ' + STATE.status.toLowerCase();
  const msgs = {
    IDLE: 'IDLE — Awaiting Tower Signal',
    MONITORING: 'MONITORING — Tower Active',
    ALERT: '🔥 SMOKE DETECTED — Alert',
    FLYING: '🚁 DRONE AIRBORNE',
    MISSION_COMPLETE: '✅ Mission Complete'
  };
  txt.textContent = msgs[STATE.status] || STATE.status;

  const timer = document.getElementById('mission-timer');
  if (timer && ['FLYING', 'MISSION_COMPLETE'].includes(STATE.status)) {
    timer.style.display = 'block';
  }
}

function goMonitoring() {
  addConsole('🏔️ AI Tower connected. Monitoring started.', 'phase');
}

function triggerAlert() {
  addConsole('🚨 ALARM! Smoke detected by tower camera!', 'danger');
  addConsole('📐 Calculating bearing to fire source...', 'warn');
  if (STATE.bearing) addConsole(`🎯 Bearing: ${STATE.bearing.toFixed(1)}° → Sector C`, 'warn');
  const sectorC = document.getElementById('sector-c');
  if (sectorC) sectorC.classList.add('has-alert');
  const idleOverlay = document.getElementById('idle-overlay');
  if (idleOverlay) idleOverlay.style.display = 'none';
  addConsole('▶ Click Sector C to expand mission view.', 'info');
}

function startFlying() {
  if (!missionStart) {
    missionStart = Date.now();
    timerInterval = setInterval(updateTimer, 1000);
  }
  addConsole('🛫 Drone takeoff initiated.', 'phase');
  addConsole('🧭 Phase A: Bearing navigation started.', 'phase');
  
  // Ensure expansion before starting animation
  if (!STATE.expanded) {
    expandSectorC();
  }
  
  const coords = document.getElementById('sc-coords');
  if (STATE.bearing && coords) coords.textContent = `Bear: ${STATE.bearing.toFixed(1)}°`;
  
  startDroneAnimation();
}

function completeMission() {
  stopDroneAnimation();
  // Make sure Sector C is expanded so fire marker is visible
  if (!STATE.expanded) expandSectorC();
  
  addConsole('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'success');
  addConsole('✅ MISSION COMPLETE — Drone landed safely.', 'success');
  addConsole('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'success');
  
  const main = document.getElementById('sector-c-main');
  const banner = document.getElementById('alert-banner');
  if (main) main.classList.add('has-alert');
  if (banner) {
    banner.style.display = 'block';
    banner.textContent = '✅ MISSION COMPLETE — Fire location confirmed 🔥';
    banner.style.background = 'linear-gradient(90deg,rgba(0,230,118,.2),rgba(0,230,118,.05))';
    banner.style.borderColor = 'rgba(0,230,118,.5)';
    banner.style.color = '#69f0ae';
    banner.style.animation = 'none';
  }

  // Show fire emoji
  const emoji = document.getElementById('sector-fire-emoji');
  if (emoji) emoji.style.display = 'block';

  if (STATE.fireLocation) {
    const fl = STATE.fireLocation;
    addConsole(`🔥 FIRE CONFIRMED at (${fl.x.toFixed(1)}, ${fl.y.toFixed(1)})`, 'danger');
    placeFireMarker(fl.x, fl.y);
  } else {
    addConsole('🔥 FIRE CONFIRMED — coordinates pending...', 'danger');
    placeFireMarker(0, 0);
    const pinText = document.getElementById('fire-pin-text');
    if (pinText) pinText.textContent = '🔥 FIRE CONFIRMED';
  }
}

function updateTimer() {
  if (!missionStart) return;
  const timer = document.getElementById('mission-timer');
  if (!timer) return;
  const s = Math.floor((Date.now() - missionStart) / 1000);
  const h = String(Math.floor(s / 3600)).padStart(2, '0');
  const m = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
  const sec = String(s % 60).padStart(2, '0');
  timer.textContent = `${h}:${m}:${sec}`;
}

// ── Sector click ─────────────────────────────────────
function onSectorClick(sector) {
  if (sector === 'c') {
    if (!STATE.expanded) expandSectorC();
  }
}

function expandSectorC() {
  STATE.expanded = true;
  document.getElementById('app').classList.add('expanded');
  document.getElementById('sector-c-main').style.display = 'block';
  document.getElementById('sector-c').style.display = 'none';
  if (STATE.status === 'ALERT' || STATE.status === 'FLYING' || STATE.status === 'MISSION_COMPLETE') {
    document.getElementById('sector-c-main').classList.add('has-alert');
  }
  const coords = document.getElementById('sc-coords');
  if (STATE.bearing && coords) coords.textContent = `Bear: ${STATE.bearing.toFixed(1)}°`;
  
  // Re-check drone animation if already flying
  if (STATE.status === 'FLYING') {
      startDroneAnimation();
  }
}

function collapseSectorC() {
  STATE.expanded = false;
  document.getElementById('app').classList.remove('expanded');
  document.getElementById('sector-c-main').style.display = 'none';
  document.getElementById('sector-c').style.display = '';
}

// ── Drone animation ───────────────────────────────────
function startDroneAnimation() {
  if (droneTimer) return; // already running
  const drone = document.getElementById('drone-svg');
  if (!drone) return;
  
  drone.style.display = 'block';
  // Wait 600ms for CSS transition to finish before reading offsetWidth
  setTimeout(() => {
    moveDroneRandom();
    if (!droneTimer) droneTimer = setInterval(moveDroneRandom, 2500);
  }, 600);
}

function moveDroneRandom() {
  const drone = document.getElementById('drone-svg');
  const parent = document.getElementById('sector-c-main');
  if (!drone || !parent || parent.style.display === 'none') return;
  
  const pw = parent.offsetWidth - 80;
  const ph = parent.offsetHeight - 80;
  if (pw <= 20 || ph <= 20) {
    // DOM not ready yet — retry once
    setTimeout(moveDroneRandom, 300);
    return;
  }
  const nx = 20 + Math.random() * pw;
  const ny = 20 + Math.random() * ph;
  drone.style.left = nx + 'px';
  drone.style.top  = ny + 'px';
}

function stopDroneAnimation() {
  if (droneTimer) { clearInterval(droneTimer); droneTimer = null; }
  const drone = document.getElementById('drone-svg');
  if (drone) drone.style.display = 'none';
}

// ── Fire marker ───────────────────────────────────────
function placeFireMarker(fx, fy) {
  const marker = document.getElementById('fire-marker');
  const parent = document.getElementById('sector-c-main');
  if (!marker || !parent) return;
  
  // Map AirSim coords to percentage (assume ±300m range)
  const px = Math.max(5, Math.min(95, 50 + (fy / 300) * 45));
  const py = Math.max(5, Math.min(90, 50 - (fx / 300) * 40));
  marker.style.left = px + '%';
  marker.style.top  = py + '%';
  marker.style.display = 'block';
  
  const pinText = document.getElementById('fire-pin-text');
  if (pinText) pinText.textContent = `🔥 FIRE (${fx.toFixed(0)}, ${fy.toFixed(0)})`;
}

// ── Console ───────────────────────────────────────────
let lastLogMsg = "";
function addConsole(msg, type) {
  const body = document.getElementById('console-body');
  if (!body) return;

  // Clean ANSI escape codes (like [2K, [35m, etc)
  let cleanMsg = msg.replace(/\x1B\[[0-9;]*[JKmsu]/g, '').trim();

  // Mask fire coordinate logs until fire is confirmed
  const fireCoordMatch = cleanMsg.match(/Fire.*?\(([\-0-9.]+),\s*([\-0-9.]+)\)/i);
  if (fireCoordMatch && STATE.status !== 'MISSION_COMPLETE') {
    // Do not reveal coordinates before confirmation
    return;
  }

  // Prevent duplicate coordinate output after confirmation
  if (fireCoordMatch && STATE.status === 'MISSION_COMPLETE' && STATE.fireReported) {
    return;
  }
  if (fireCoordMatch && STATE.status === 'MISSION_COMPLETE') {
    STATE.fireReported = true;
  }

  
  // Deduplicate consecutive identical messages
  if (cleanMsg === lastLogMsg) return;
  lastLogMsg = cleanMsg;

  const cls  = classifyLog(cleanMsg, type);
  const ts   = new Date().toLocaleTimeString('tr', { hour12: false });
  
  // Avoid double timestamping if the message already starts with one
  const displayMsg = cleanMsg.startsWith('[') ? cleanMsg : `[${ts}] ${cleanMsg}`;
  
  const line = document.createElement('div');
  line.className = 'log-line ' + cls;
  line.textContent = displayMsg;
  body.appendChild(line);
  
  // Keep last 200 lines
  while (body.children.length > 200) body.removeChild(body.firstChild);
  body.scrollTop = body.scrollHeight;
}

function classifyLog(msg, hint) {
  if (hint) return hint;
  if (/Pos:|Alt:|Step|CTE/i.test(msg)) return 'pos';
  if (/ALARM|SMOKE|FIRE|🚨|🔥/i.test(msg)) return 'danger';
  if (/PHASE|bearing|dispatch|PPO|Training/i.test(msg)) return 'phase';
  if (/Takeoff|airborne|climb|Landing/i.test(msg)) return 'warn';
  if (/complete|confirmed|saved|✅/i.test(msg)) return 'success';
  return 'info';
}

// ── Gallery ───────────────────────────────────────────
function switchTab(tab) {
  STATE.activeTab = tab;
  document.querySelectorAll('.gtab').forEach(t => t.classList.remove('active'));
  const tabEl = document.getElementById('gtab-' + tab);
  if (tabEl) tabEl.classList.add('active');
  renderGallery();
}

function renderGallery() {
  const body  = document.getElementById('gallery-body');
  if (!body) return;
  const items = STATE.photos[STATE.activeTab] || [];
  if (!items.length) {
    body.innerHTML = '<div class="empty-gallery">📡 No images yet</div>';
    return;
  }
  const grid = document.createElement('div');
  grid.className = 'gallery-grid';
  items.forEach((ph, i) => {
    const d = document.createElement('div');
    d.className = 'thumb';
    if (ph.kind === 'json') {
      d.innerHTML = `<div style="height:100%;display:flex;align-items:center;justify-content:center;font-size:20px">📋</div><div class="thumb-label">${ph.name}</div>`;
    } else {
      d.innerHTML = `<img src="${ph.url}" loading="lazy" alt="${ph.name}"/><div class="thumb-label">${ph.name}</div>`;
    }
    d.onclick = () => openLightbox(STATE.activeTab, i);
    grid.appendChild(d);
  });
  body.innerHTML = '';
  body.appendChild(grid);
}

// ── Lightbox ──────────────────────────────────────────
function openLightbox(tab, idx) {
  lbPhotos = (STATE.photos[tab] || []).filter(p => p.kind !== 'json');
  if (!lbPhotos.length) return;
  lbIdx = Math.min(idx, lbPhotos.length - 1);
  showLbPhoto();
  document.getElementById('lightbox').classList.add('open');
}

function showLbPhoto() {
  const ph = lbPhotos[lbIdx];
  const img = document.getElementById('lb-img');
  const caption = document.getElementById('lb-caption');
  const counter = document.getElementById('lb-counter');
  if (img) img.src = ph.url;
  if (caption) caption.textContent = ph.name;
  if (counter) counter.textContent = `${lbIdx + 1} / ${lbPhotos.length}`;
}

function lbNav(dir) {
  lbIdx = (lbIdx + dir + lbPhotos.length) % lbPhotos.length;
  showLbPhoto();
}

function closeLightbox() {
  const lb = document.getElementById('lightbox');
  if (lb) lb.classList.remove('open');
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLightbox();
  if (e.key === 'ArrowLeft')  lbNav(-1);
  if (e.key === 'ArrowRight') lbNav(1);
});

// ── Output folder ─────────────────────────────────────
function openOutputFolder() {
  const path = '../autonomousflight/output';
  addConsole(`📁 Output folder: ${path}`, 'info');
  try {
    window.open('file:///' + window.location.hostname.replace('localhost', '') + '/../autonomousflight/output', '_blank');
  } catch (e) {}
}
