// EasyPerception — 三栏 Dashboard

// === State ===
let drawnZone = null;
let isDrawing = false;
let drawStart = null;
let eventsWs = null;
let isMonitoring = false;
let eventCount = 0;
let alertCount = 0;
let lastRawJson = null;
let jsonExpanded = false;

// === 事件类型中文映射 ===
const EVENT_TYPE_LABELS = {
  zone_entry: '进入区域',
  object_entered: '物体出现',
  object_left: '物体离开',
  object_removed: '物体移除',
  dwell_warning: '停留警告',
  dwell_alert: '停留告警',
  dwell_critical: '停留严重',
  new_class: '新类型',
  class_gone: '类型消失',
  scene_start: '场景初始化',
  suspicious_removal: '异常移除',
  monitoring_started: '开始监控',
  monitoring_stopped: '停止监控',
};

const CLASS_LABELS = {
  bottle: '水瓶', cup: '杯子', 'cell phone': '手机', laptop: '笔记本',
  book: '书', remote: '遥控器', keyboard: '键盘', person: '人',
  mouse: '鼠标', chair: '椅子',
};

function eventTypeLabel(type) { return EVENT_TYPE_LABELS[type] || type; }
function classLabel(cls) { return CLASS_LABELS[cls] || cls; }

// === Canvas: Zone Drawing (auto-start on release) ===

function initCanvas() {
  const canvas = document.getElementById('zoneCanvas');
  const videoEl = document.getElementById('videoFeed');

  function syncSize() {
    canvas.width = videoEl.clientWidth;
    canvas.height = videoEl.clientHeight;
    if (drawnZone) redrawZone();
  }
  syncSize();

  canvas.onmousedown = (e) => {
    if (isMonitoring) return;
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
    drawZonePreview(drawStart, {
      x: (e.clientX - rect.left) / rect.width,
      y: (e.clientY - rect.top) / rect.height,
    });
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
    // Only start if zone is big enough (not accidental click)
    const w = drawnZone.x2 - drawnZone.x1;
    const h = drawnZone.y2 - drawnZone.y1;
    if (w > 0.05 && h > 0.05) {
      autoStartMonitoring();
    } else {
      drawnZone = null;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  window.addEventListener('resize', syncSize);
}

function drawZonePreview(start, end) {
  const canvas = document.getElementById('zoneCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const x = Math.min(start.x, end.x) * canvas.width;
  const y = Math.min(start.y, end.y) * canvas.height;
  const w = Math.abs(end.x - start.x) * canvas.width;
  const h = Math.abs(end.y - start.y) * canvas.height;

  ctx.strokeStyle = isMonitoring ? '#00d4aa' : '#3b82f6';
  ctx.lineWidth = 2;
  ctx.setLineDash(isMonitoring ? [] : [6, 4]);
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = isMonitoring ? 'rgba(0, 212, 170, 0.06)' : 'rgba(59, 130, 246, 0.06)';
  ctx.fillRect(x, y, w, h);

  ctx.font = '12px Inter, sans-serif';
  ctx.fillStyle = isMonitoring ? '#00d4aa' : '#3b82f6';
  ctx.setLineDash([]);
  ctx.fillText(isMonitoring ? '监控区域' : '选择中...', x + 4, y - 6);
}

function redrawZone() {
  if (!drawnZone) return;
  drawZonePreview(
    { x: drawnZone.x1, y: drawnZone.y1 },
    { x: drawnZone.x2, y: drawnZone.y2 }
  );
}

// === Monitoring (auto-start, one-click clear) ===

async function autoStartMonitoring() {
  if (!drawnZone) return;

  try {
    await fetch('/api/watchpoint', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'clear' }),
    });

    const res = await fetch('/api/zone/setup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        zone: drawnZone,
        target_class: 'any',
        label: '监控区',
      }),
    });

    if (res.ok) {
      isMonitoring = true;
      document.getElementById('drawHint').classList.add('hidden');
      document.getElementById('zoneStatus').classList.remove('hidden');
      redrawZone();
      addEvent({
        time_str: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
        event_type: 'monitoring_started',
        severity: 'info',
        description: '已开始监控选定区域内的所有物品',
      });
    }
  } catch (e) {
    console.error('启动监控失败:', e);
  }
}

async function clearZone() {
  try {
    await fetch('/api/watchpoint', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'clear' }),
    });
  } catch (e) { /* ignore */ }

  isMonitoring = false;
  drawnZone = null;
  document.getElementById('drawHint').classList.remove('hidden');
  document.getElementById('zoneStatus').classList.add('hidden');

  const canvas = document.getElementById('zoneCanvas');
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

  resetLLMPanel();
  addEvent({
    time_str: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
    event_type: 'monitoring_stopped',
    severity: 'info',
    description: '监控已停止',
  });
}

// === Events WebSocket ===

function connectEventsWs() {
  if (eventsWs && eventsWs.readyState <= 1) return;

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  eventsWs = new WebSocket(`${protocol}//${location.host}/ws/events`);

  eventsWs.onopen = () => {
    document.querySelector('.header-status .status-dot').className = 'status-dot connected';
    document.querySelector('.header-status span').textContent = '已连接';
  };

  eventsWs.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'event') handleEvent(msg.data);
    else if (msg.type === 'llm_update') handleLLMUpdate(msg.data);
    else if (msg.type === 'dwell_update') handleDwellUpdate(msg.data);
    else if (msg.type === 'stats_update') handleStatsUpdate(msg.data);
    else if (msg.type === 'raw_json') handleRawJson(msg.data);
  };

  eventsWs.onclose = () => {
    document.querySelector('.header-status .status-dot').className = 'status-dot error';
    document.querySelector('.header-status span').textContent = '已断开';
    setTimeout(connectEventsWs, 3000);
  };
}

// === Event Handling ===

function handleEvent(data) {
  addEvent(data);
  if (data.severity === 'alert' || data.severity === 'critical') {
    playAlertSound();
    alertCount++;
    document.getElementById('statAlerts').textContent = alertCount;
  }
}

function addEvent(data, silent) {
  const timeline = document.getElementById('eventTimeline');
  const empty = timeline.querySelector('.event-empty');
  if (empty) empty.remove();

  const card = document.createElement('div');
  card.className = `event-card severity-${data.severity || 'info'}`;
  const typeLabel = eventTypeLabel(data.event_type);

  card.innerHTML = `
    <div class="event-time">${data.time_str || '--:--:--'}</div>
    <div class="event-body">
      <div class="event-type ${data.severity || 'info'}">${escapeHtml(typeLabel)}</div>
      <div class="event-desc">${escapeHtml(data.description || '')}</div>
    </div>
  `;

  timeline.insertBefore(card, timeline.firstChild);
  while (timeline.children.length > 50) timeline.removeChild(timeline.lastChild);

  if (!silent) {
    eventCount++;
    document.getElementById('statEvents').textContent = eventCount;
  }
}

// === JSON Preview ===

function handleRawJson(data) {
  lastRawJson = data;
  if (jsonExpanded) updateJsonPreview();
}

function toggleJsonPreview() {
  jsonExpanded = !jsonExpanded;
  document.getElementById('jsonArrow').textContent = jsonExpanded ? '▼' : '▶';
  document.getElementById('jsonPreview').classList.toggle('hidden', !jsonExpanded);
  if (jsonExpanded) updateJsonPreview();
}

function updateJsonPreview() {
  const el = document.getElementById('jsonPreview');
  if (lastRawJson) {
    el.textContent = JSON.stringify(lastRawJson, null, 2);
  }
}

// === LLM Updates ===

function handleLLMUpdate(data) {
  const alertBox = document.getElementById('llmAlertBox');
  const narrative = document.getElementById('llmNarrative');
  const actions = document.getElementById('llmActions');
  const threatDot = document.querySelector('.threat-dot');
  const threatText = document.getElementById('threatText');

  alertBox.textContent = data.alert_text || '';
  alertBox.className = 'llm-alert-box';
  if (data.threat_level === 'attention') alertBox.classList.add('attention');
  else if (data.threat_level === 'warning') alertBox.classList.add('warning');
  else if (data.threat_level === 'alert') alertBox.classList.add('alert');

  narrative.textContent = data.narrative || '';

  actions.innerHTML = '';
  const actionList = data.actions || [];
  if (actionList.length === 0) {
    actions.innerHTML = '<div class="llm-placeholder">暂无建议</div>';
  } else {
    actionList.forEach(action => {
      const item = document.createElement('div');
      item.className = 'action-item';
      item.innerHTML = `<span class="action-bullet">→</span><span>${escapeHtml(action)}</span>`;
      actions.appendChild(item);
    });
  }

  const level = data.threat_level || 'normal';
  threatDot.className = `threat-dot threat-${level}`;
  const levelNames = { normal: '正常', attention: '关注', warning: '警告', alert: '告警' };
  threatText.textContent = levelNames[level] || level;
}

function resetLLMPanel() {
  document.getElementById('llmAlertBox').innerHTML = '<div class="llm-placeholder">检测到事件后 LLM 将自动分析...</div>';
  document.getElementById('llmAlertBox').className = 'llm-alert-box';
  document.getElementById('llmNarrative').innerHTML = '<div class="llm-placeholder">等待事件分析...</div>';
  document.getElementById('llmActions').innerHTML = '<div class="llm-placeholder">暂无建议</div>';
  document.querySelector('.threat-dot').className = 'threat-dot threat-normal';
  document.getElementById('threatText').textContent = '正常';
}

// === Dwell Updates ===

function handleDwellUpdate(data) {
  const section = document.getElementById('dwellSection');
  const container = document.getElementById('dwellTimers');
  const items = [];
  for (const [, objects] of Object.entries(data.dwell_times || {})) {
    for (const obj of objects) items.push(obj);
  }

  if (items.length === 0) {
    section.style.display = 'none';
    document.getElementById('statDwell').textContent = '-';
    return;
  }

  section.style.display = '';
  container.innerHTML = '';
  let maxDwell = 0;

  items.forEach(item => {
    const dwell = item.dwell_sec || 0;
    if (dwell > maxDwell) maxDwell = dwell;

    let color = 'var(--success)';
    if (dwell >= 120) color = 'var(--critical)';
    else if (dwell >= 60) color = 'var(--danger)';
    else if (dwell >= 30) color = 'var(--warning)';

    const pct = Math.min(100, (dwell / 120) * 100);
    const timer = document.createElement('div');
    timer.className = 'dwell-timer';
    timer.innerHTML = `
      <div class="dwell-timer-label">${escapeHtml(classLabel(item.class))}</div>
      <div class="dwell-bar"><div class="dwell-bar-fill" style="width:${pct}%;background:${color}"></div></div>
      <div class="dwell-timer-value" style="color:${color}">${dwell.toFixed(0)}s</div>
    `;
    container.appendChild(timer);
  });

  document.getElementById('statDwell').textContent = `${maxDwell.toFixed(0)}s`;
}

function handleStatsUpdate(data) {
  if (data.tracks !== undefined) {
    document.getElementById('statTracks').textContent = data.tracks;
  }
}

// === LLM Chat (enhanced — context-aware) ===

async function sendLLMQuery() {
  const input = document.getElementById('llmInput');
  const query = input.value.trim();
  if (!query) return;
  input.value = '';

  addChatMsg(query, 'user');
  const loadingId = addChatMsg('<span class="spinner"></span>分析中...', 'llm', true);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: query }),
    });
    removeChatMsg(loadingId);
    if (res.ok) {
      const data = await res.json();
      addChatMsg(data.answer || '无法回答', 'llm');
    } else {
      addChatMsg('服务暂时不可用', 'llm');
    }
  } catch (e) {
    removeChatMsg(loadingId);
    addChatMsg('连接失败: ' + e.message, 'llm');
  }
}

let chatMsgId = 0;
function addChatMsg(content, type, isHtml) {
  const id = 'chat-' + (++chatMsgId);
  const history = document.getElementById('chatHistory');
  const div = document.createElement('div');
  div.className = `chat-msg chat-msg-${type}`;
  div.id = id;
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble';
  if (isHtml) bubble.innerHTML = content;
  else bubble.textContent = content;
  div.appendChild(bubble);
  history.appendChild(div);
  history.scrollTop = history.scrollHeight;
  return id;
}

function removeChatMsg(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// === Health Polling ===

async function pollHealth() {
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const data = await res.json();
      if (data.connected) {
        document.getElementById('footerFPS').textContent = `帧率: ${data.fps || '--'}`;
        document.getElementById('footerLatency').textContent = `延迟: ${data.latency_ms || '--'}ms`;
        document.getElementById('footerPipeline').textContent = `管线: ${data.state || '--'}`;
      }
    }
  } catch (e) { /* ignore */ }
}

// === Init ===

async function loadInitialState() {
  // Clean slate on every page load
  try { await fetch('/api/reset', { method: 'POST' }); } catch (e) { /* ignore */ }
  eventCount = 0;
  alertCount = 0;
}

function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
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
  } catch (e) { /* ignore */ }
}

document.addEventListener('DOMContentLoaded', () => {
  initCanvas();
  connectEventsWs();
  loadInitialState();
  setInterval(pollHealth, 5000);
  pollHealth();
});
