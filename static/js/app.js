/* SignSpeak — client: kirim frame webcam ke backend, tampilkan ejaan + audio. */
const CLIENT_ID = "client-" + Math.random().toString(36).slice(2, 10);
const HEADERS = { "Content-Type": "application/json", "X-Client-Id": CLIENT_ID };
const INTERVAL_MS = 90;          // jarak antar frame ke server

const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay");
const ctx = overlay.getContext("2d");

let stream = null, timer = null, running = false, lastFrameAt = 0;
let busy = false, currentText = "", historyItems = [];

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17]
];

async function loadStatus() {
  const el = $("modelStatus");
  try {
    const r = await fetch("/api/status");
    const s = await r.json();
    const okAsl = s.asl.loaded ? "Model ASL siap" : "Model ASL: " + (s.asl.error || "belum dimuat");
    const ttsName = s.tts.using_finetuned ? "TTS LJSpeech" : "TTS dasar (microsoft)";
    const okTts = s.tts.loaded ? ttsName : ttsName + " — muat saat bicara pertama";
    el.textContent = `${okAsl} · ${okTts} · ${s.asl.num_classes} kelas`;
    el.className = "status ok";
  } catch (e) {
    el.textContent = "Gagal menghubungi server: " + e.message;
    el.className = "status err";
  }
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" }, audio: false
    });
  } catch (e) {
    alert("Tidak dapat mengakses kamera: " + e.message);
    return;
  }
  video.srcObject = stream;
  await video.play();
  running = true;
  $("btnStart").disabled = true;
  $("btnStop").disabled = false;
  $("btnSpeak").disabled = false;
  resizeCanvas();
  loop();
}

function stopCamera() {
  running = false;
  clearTimeout(timer);
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  $("btnStart").disabled = false;
  $("btnStop").disabled = true;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function resizeCanvas() {
  overlay.width = video.videoWidth || 640;
  overlay.height = video.videoHeight || 480;
}

function drawOverlay(data) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const sx = overlay.width / (data.frame_size?.w || overlay.width);
  const sy = overlay.height / (data.frame_size?.h || overlay.height);
  if (data.bbox) {
    const [x0, y0, x1, y1] = data.bbox;
    ctx.strokeStyle = "#4f8cff";
    ctx.lineWidth = 3;
    ctx.strokeRect(x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy);
    if (data.label) {
      const label = `${data.label} ${(data.confidence * 100).toFixed(0)}%`;
      ctx.font = "600 18px sans-serif";
      const w = ctx.measureText(label).width + 14;
      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.fillRect(x0 * sx, Math.max(0, y0 * sy - 26), w, 24);
      ctx.fillStyle = "#eaf0ff";
      ctx.fillText(label, x0 * sx + 7, Math.max(16, y0 * sy - 8));
    }
  }
  if (data.landmarks?.length) {
    ctx.fillStyle = "#22d3a7";
    for (const p of data.landmarks) {
      ctx.beginPath();
      ctx.arc(p.x * sx, p.y * sy, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.lineWidth = 1.5;
    for (const [a, b] of HAND_CONNECTIONS) {
      const p = data.landmarks[a], q = data.landmarks[b];
      if (!p || !q) continue;
      ctx.beginPath();
      ctx.moveTo(p.x * sx, p.y * sy);
      ctx.lineTo(q.x * sx, q.y * sy);
      ctx.stroke();
    }
  }
}

function renderText(text) {
  const box = $("textOut");
  box.innerHTML = "";
  if (!text) {
    box.innerHTML = '<span class="placeholder">Belum ada teks…</span>';
  } else {
    box.textContent = text;
    const cur = document.createElement("span");
    cur.className = "cursor";
    box.appendChild(cur);
  }
  const words = text.trim() ? text.trim().split(/\s+/) : [];
  $("words").innerHTML = words.map((w) => `<span class="word">${w}</span>`).join("");
  $("btnSpeak").disabled = !text.trim();
}

function renderPrediction(data) {
  const spell = data.spell || {};
  $("currentLetter").textContent = data.hand_detected ? (data.label || "?") : "–";
  const pct = (data.confidence || 0) * 100;
  $("confFill").style.width = pct.toFixed(0) + "%";
  $("confText").textContent = pct.toFixed(0) + "%";
  $("top3").innerHTML = (data.top || [])
    .map((t) => `<span>${t.label} ${(t.confidence * 100).toFixed(0)}%</span>`)
    .join("");

  // status gerak tangan (motion gate)
  const settled = spell.motion ? spell.motion.settled : true;
  const mEl = $("motionState");
  mEl.textContent = settled ? "diam" : "bergerak";
  mEl.className = settled ? "motion" : "motion moving";

  // progres kestabilan huruf
  const p = spell.progress || {};
  const ratio = Math.max(0, Math.min(1, p.ratio || 0));
  const holdBox = document.querySelector(".hold");
  holdBox.classList.toggle("ready", !!p.ready);
  $("holdFill").style.width = (ratio * 100).toFixed(0) + "%";
  $("holdLabel").textContent = p.candidate
    ? `${p.candidate} ${p.votes}/${p.window}`
    : `${p.filled || 0}/${p.window || 0}`;
  if (!data.hand_detected) {
    $("holdHint").textContent = "Tangan tidak terlihat — masukkan tangan ke kamera.";
  } else if (!settled) {
    $("holdHint").textContent = "Tangan sedang berpindah — tunggu sampai diam untuk huruf berikutnya.";
  } else if (spell.last_committed && p.candidate === spell.last_committed) {
    $("holdHint").textContent = `Tahan lebih lama untuk huruf dobel (${spell.last_committed}${spell.last_committed}).`;
  } else if (p.ready) {
    $("holdHint").textContent = "Siap! Pindahkan tangan ke huruf berikutnya.";
  } else {
    $("holdHint").textContent = "Tahan pose huruf sampai bar penuh, lalu pindah ke huruf berikutnya.";
  }
}

function addHistory(text, audioB64) {
  const list = $("history");
  list.querySelector(".empty")?.remove();
  const bin = atob(audioB64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const blob = new Blob([bytes], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);

  const li = document.createElement("li");
  li.className = "item";
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.src = url;
  const info = document.createElement("div");
  info.innerHTML = `<div class="txt">${text}</div><div class="sub">${new Date().toLocaleTimeString()}</div>`;
  const dl = document.createElement("a");
  dl.className = "dl";
  dl.href = url;
  dl.download = `signspeak-${Date.now()}.wav`;
  dl.textContent = "unduh .wav";
  li.append(info, audio, dl);
  list.prepend(li);
  historyItems.push(url);
  if (historyItems.length > 8) URL.revokeObjectURL(historyItems.shift());
  audio.play().catch(() => {});
}

async function sendFrame() {
  if (!running) return;
  const cv = document.createElement("canvas");
  const w = 480, h = Math.round((video.videoHeight / video.videoWidth) * w) || 360;
  cv.width = w; cv.height = h;
  cv.getContext("2d").drawImage(video, 0, 0, w, h);
  const image = cv.toDataURL("image/jpeg", 0.7);
  try {
    const r = await fetch("/api/detect", {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({ image, speak: $("autoSpeak").checked }),
    });
    if (r.ok) {
      const data = await r.json();
      lastFrameAt = Date.now();
      resizeCanvas();
      drawOverlay(data);
      renderPrediction(data);
      currentText = data.spell?.text || "";
      renderText(currentText);
      if (data.audio) addHistory(data.spoken_text || currentText, data.audio);
    }
  } catch (e) {
    /* server sedang sibuk / mati: coba lagi di frame berikutnya */
  } finally {
    busy = false;
  }
}

function loop() {
  if (!running) return;
  if (!busy) {
    busy = true;
    sendFrame();
  }
  timer = setTimeout(loop, INTERVAL_MS);
}

async function speakNow() {
  const text = currentText.trim();
  if (!text) return;
  $("btnSpeak").disabled = true;
  try {
    const r = await fetch("/api/speak", {
      method: "POST", headers: HEADERS,
      body: JSON.stringify({ text, speaker: $("speaker").value }),
    });
    if (r.ok) addHistory(text, (await r.json()).audio);
    await fetch("/api/reset", { method: "POST", headers: HEADERS });
    currentText = "";
    renderText("");
  } finally {
    $("btnSpeak").disabled = !currentText.trim();
  }
}

async function clearText() {
  currentText = "";
  renderText("");
  await fetch("/api/reset", { method: "POST", headers: HEADERS });
}

async function pushSettings() {
  $("winVal").textContent = $("win").value;
  $("agrVal").textContent = Number($("agr").value).toFixed(2);
  $("cdVal").textContent = $("cd").value;
  $("rpVal").textContent = $("rp").value;
  $("idlVal").textContent = $("idl").value;
  await fetch("/api/settings", {
    method: "POST", headers: HEADERS,
    body: JSON.stringify({
      window: +$("win").value,
      agreement: +$("agr").value,
      cooldown: +$("cd").value,
      repeat_hold_ms: +$("rp").value,
      speak_mode: $("speakMode").value,
      word_idle_ms: +$("idl").value,
      speaker: $("speaker").value,
    }),
  });
}

["win", "agr", "cd", "rp", "idl", "speakMode", "speaker"].forEach((id) =>
  $(id).addEventListener("change", pushSettings)
);
pushSettings();
$("btnStart").addEventListener("click", startCamera);
$("btnStop").addEventListener("click", stopCamera);
$("btnSpeak").addEventListener("click", speakNow);
$("btnClear").addEventListener("click", clearText);
window.addEventListener("beforeunload", stopCamera);

loadStatus();
setInterval(loadStatus, 30000);
