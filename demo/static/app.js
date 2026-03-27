// EasyPerception Investor Demo — Frontend Logic

// === State ===
let currentTab = 'farm';
let isWatching = false;
let drawnZone = null;   // {x1, y1, x2, y2} normalized
let isDrawing = false;
let drawStart = null;
let alertWs = null;
let zoneId = null;

// === Tab Switching ===
function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab-btn').forEach((btn, i) => {
    btn.classList.toggle('active', (i === 0 && tab === 'farm') || (i === 1 && tab === 'watch'));
  });
  document.getElementById('farmPanel').classList.toggle('hidden', tab !== 'farm');
  document.getElementById('watchPanel').classList.toggle('hidden', tab !== 'watch');
  document.getElementById('zoneCanvas').classList.toggle('hidden', tab !== 'watch');
  document.getElementById('drawHint').classList.toggle('hidden', tab !== 'watch' || isWatching);

  if (tab === 'watch') {
    initCanvas();
    connectAlertWs();
  }
}

// === Demo 1: Farm Chat ===
async function sendQuery(text) {
  const input = document.getElementById('chatInput');
  const query = text || input.value.trim();
  if (!query) return;

  input.value = '';
  addMessage(query, 'user');

  const sendBtn = document.getElementById('sendBtn');
  sendBtn.disabled = true;

  // Show loading
  const loadingId = addMessage('<span class="spinner"></span>Analyzing...', 'system', true);

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: query }),
    });
    if (!res.ok) {
      const errText = await res.text();
      removeMessage(loadingId);
      addMessage('Server error: ' + errText, 'system');
      sendBtn.disabled = false;
      return;
    }
    const data = await res.json();
    removeMessage(loadingId);

    if (data.error) {
      addMessage('Error: ' + data.error, 'system');
    } else {
      let html = data.answer;
      if (data.scene_data && data.scene_data.scene) {
        const s = data.scene_data.scene;
        html += `<div class="data-card">` +
          `Objects: ${s.object_count || 0} | ` +
          `Classes: ${(s.classes_present || []).join(', ') || 'none'} | ` +
          `Risk: ${s.risk_level || 'clear'}` +
          `</div>`;
      }
      if (data.actions_taken && data.actions_taken.length > 0) {
        html += `<div class="data-card">${data.actions_taken.join('<br>')}</div>`;
      }
      addMessage(html, 'system', true);
    }
  } catch (e) {
    removeMessage(loadingId);
    addMessage('Connection error: ' + e.message, 'system');
  }

  sendBtn.disabled = false;
}

let messageCounter = 0;
function addMessage(content, type, isHtml = false) {
  const id = 'msg-' + (++messageCounter);
  const container = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'message ' + type;
  div.id = id;
  if (isHtml) {
    div.innerHTML = content;
  } else {
    div.textContent = content;
  }
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return id;
}

function removeMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// === Demo 2: Conversational Watchpoint ===
function initCanvas() {
  const canvas = document.getElementById('zoneCanvas');
  const videoEl = document.getElementById('videoFeed');

  canvas.width = videoEl.clientWidth;
  canvas.height = videoEl.clientHeight;

  canvas.onmousedown = (e) => {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    drawStart = {
      x: (e.clientX - rect.left) / rect.width,
      y: (e.clientY - rect.top) / rect.height,
    };
  };

  canvas.onmousemove = (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const current = {
      x: (e.clientX - rect.left) / rect.width,
      y: (e.clientY - rect.top) / rect.height,
    };
    drawZonePreview(drawStart, current);
  };

  canvas.onmouseup = (e) => {
    if (!isDrawing) return;
    isDrawing = false;
    const rect = canvas.getBoundingClientRect();
    const end = {
      x: (e.clientX - rect.left) / rect.width,
      y: (e.clientY - rect.top) / rect.height,
    };
    drawnZone = {
      x1: Math.min(drawStart.x, end.x),
      y1: Math.min(drawStart.y, end.y),
      x2: Math.max(drawStart.x, end.x),
      y2: Math.max(drawStart.y, end.y),
    };
    drawZonePreview(drawStart, end);
    addWatchMessage('Area selected.', 'system', true);
  };

  window.addEventListener('resize', () => {
    canvas.width = videoEl.clientWidth;
    canvas.height = videoEl.clientHeight;
    if (drawnZone) redrawZone();
  });
}

function drawZonePreview(start, end) {
  const canvas = document.getElementById('zoneCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const x = Math.min(start.x, end.x) * canvas.width;
  const y = Math.min(start.y, end.y) * canvas.height;
  const w = Math.abs(end.x - start.x) * canvas.width;
  const h = Math.abs(end.y - start.y) * canvas.height;

  ctx.strokeStyle = '#00d4aa';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = 'rgba(0, 212, 170, 0.1)';
  ctx.fillRect(x, y, w, h);

  ctx.font = '14px Inter, sans-serif';
  ctx.fillStyle = '#00d4aa';
  ctx.setLineDash([]);
  ctx.fillText('Selected Zone', x + 4, y - 6);
}

function redrawZone() {
  if (!drawnZone) return;
  drawZonePreview(
    { x: drawnZone.x1, y: drawnZone.y1 },
    { x: drawnZone.x2, y: drawnZone.y2 }
  );
}

// Watch chat messages
let watchMsgCounter = 0;
function addWatchMessage(content, type, isHtml = false) {
  const id = 'wmsg-' + (++watchMsgCounter);
  const container = document.getElementById('watchMessages');
  const div = document.createElement('div');
  div.className = 'message ' + type;
  div.id = id;
  if (isHtml) {
    div.innerHTML = content;
  } else {
    div.textContent = content;
  }
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return id;
}

function removeWatchMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

async function sendWatchCommand(text) {
  const input = document.getElementById('watchInput');
  const query = text || input.value.trim();
  if (!query) return;

  input.value = '';
  addWatchMessage(query, 'user');

  const btn = document.getElementById('watchSendBtn');
  btn.disabled = true;

  const loadingId = addWatchMessage('<span class="spinner"></span>Processing...', 'system', true);

  try {
    const res = await fetch('/api/watch-command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: query, zone: drawnZone }),
    });
    if (!res.ok) {
      removeWatchMessage(loadingId);
      addWatchMessage('Server error. Please try again.', 'system');
      btn.disabled = false;
      return;
    }
    const data = await res.json();
    removeWatchMessage(loadingId);

    // Render response with markdown bold
    const answer = (data.answer || '').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    addWatchMessage(answer, 'system', true);

    // Update status based on action
    if (data.action === 'watching') {
      isWatching = true;
      document.getElementById('statusDot').className = 'status-dot monitoring';
      document.getElementById('statusText').textContent = 'Monitoring: ' + (data.target_class || 'person');
      document.getElementById('drawHint').classList.add('hidden');
      // Clear canvas — server draws the zone overlay now
      const canvas = document.getElementById('zoneCanvas');
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    } else if (data.action === 'stopped') {
      isWatching = false;
      drawnZone = null;
      document.getElementById('statusDot').className = 'status-dot';
      document.getElementById('statusText').textContent = 'Ready';
      document.getElementById('drawHint').classList.remove('hidden');
    } else if (data.action === 'need_zone') {
      document.getElementById('drawHint').classList.remove('hidden');
    }
  } catch (e) {
    removeWatchMessage(loadingId);
    addWatchMessage('Connection error: ' + e.message, 'system');
  }

  btn.disabled = false;
}

// === Alert WebSocket ===
function connectAlertWs() {
  if (alertWs && alertWs.readyState <= 1) return;

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  alertWs = new WebSocket(`${protocol}//${location.host}/ws/alerts`);

  alertWs.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'alert') {
      onAlert(msg.data);
    }
  };

  alertWs.onclose = () => {
    setTimeout(connectAlertWs, 3000);
  };
}

function onAlert(data) {
  // Update status
  document.getElementById('statusDot').className = 'status-dot alert';
  document.getElementById('statusText').textContent = 'ALERT!';

  // Sound
  playAlertSound();

  // Add alert as a chat message
  const alertHtml = `<div class="alert-item" style="margin:0;">
    <div class="time">${data.time_str}</div>
    <div class="msg">${data.message}</div>
  </div>`;
  addWatchMessage(alertHtml, 'system', true);

  // Reset status after 3s
  setTimeout(() => {
    if (isWatching) {
      document.getElementById('statusDot').className = 'status-dot monitoring';
      document.getElementById('statusText').textContent = 'Monitoring...';
    }
  }, 3000);
}

function playAlertSound() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.frequency.value = 880;
    osc.type = 'sine';
    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.5);
  } catch (e) {
    // Audio not available
  }
}

// === Init ===
document.addEventListener('DOMContentLoaded', () => {
  // Pre-connect alert WS
  connectAlertWs();
});
