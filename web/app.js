/* ==========================================================================
   FreeSight-OS v7.0 Real-Time Studio Clientside Controller
   Real-Time Telemetry Streaming, Floating Reticle & Interactive Gaze Sandbox
   ========================================================================== */

(function () {
  'use strict';

  // --- Audio Synthesizer (Web Audio API) ---
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

  function playTone(freq, type = 'sine', duration = 0.08) {
    try {
      if (audioCtx.state === 'suspended') audioCtx.resume();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
      gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
      // Audio autoplay policy guard
    }
  }

  function playClickSound() {
    playTone(1200, 'triangle', 0.06);
    setTimeout(() => playTone(1800, 'sine', 0.08), 30);
  }

  // --- DOM Elements ---
  const gazePointer = document.getElementById('gaze-pointer');
  const dwellBar = document.getElementById('dwell-bar');
  const gazeLabel = document.getElementById('gaze-label');

  const statLatency = document.getElementById('stat-latency');
  const statFps = document.getElementById('stat-fps');
  const statConf = document.getElementById('stat-conf');
  const statCb = document.getElementById('stat-cb');
  const statScrollVel = document.getElementById('stat-scroll-vel');
  const statWatchdog = document.getElementById('stat-watchdog');

  const earValue = document.getElementById('ear-value');
  const earBar = document.getElementById('ear-bar');
  const pupilCoords = document.getElementById('pupil-coords');
  const pupilCrosshair = document.getElementById('pupil-crosshair');
  const headPoseDeg = document.getElementById('head-pose-deg');
  const yawBadge = document.getElementById('yaw-badge');
  const pitchBadge = document.getElementById('pitch-badge');
  const rollBadge = document.getElementById('roll-badge');

  const dirArrow = document.getElementById('dir-arrow');
  const dirText = document.getElementById('dir-text');
  const scrollStatus = document.getElementById('scroll-status');
  const scrollContent = document.getElementById('scroll-content');
  const activityLog = document.getElementById('activity-log');

  const modeScrollBtn = document.getElementById('mode-scroll-btn');
  const modeClickBtn = document.getElementById('mode-click-btn');
  const toggleOverlayBtn = document.getElementById('toggle-overlay-btn');
  const cameraReconnectBtn = document.getElementById('camera-reconnect-btn');
  const cameraShutdownBtn = document.getElementById('camera-shutdown-btn');
  const clearLogBtn = document.getElementById('clear-log-btn');

  // --- Dwell & Tracking State ---
  let currentTarget = null;
  let dwellStartTime = 0;
  const DWELL_THRESHOLD_MS = 600;
  let lastClickTime = 0;
  let currentMode = 'directional_scroll';
  let showOverlay = true;

  // Smoothing filter for pointer
  let reticleX = window.innerWidth / 2;
  let reticleY = window.innerHeight / 2;

  // --- Logging Helper ---
  function appendLog(text, colorClass = 'text-cyan') {
    const line = document.createElement('div');
    line.className = `log-line ${colorClass}`;
    const timestamp = new Date().toLocaleTimeString();
    line.textContent = `[${timestamp}] ${text}`;
    activityLog.appendChild(line);
    activityLog.scrollTop = activityLog.scrollHeight;

    while (activityLog.children.length > 30) {
      activityLog.removeChild(activityLog.firstChild);
    }
  }

  // --- Target Click Handler ---
  function triggerTargetClick(targetEl) {
    const now = Date.now();
    if (now - lastClickTime < 400) return;
    lastClickTime = now;

    playClickSound();
    targetEl.classList.add('clicked-flash');
    setTimeout(() => targetEl.classList.remove('clicked-flash'), 400);

    const countEl = targetEl.querySelector('span[id^="count-"]');
    if (countEl) {
      countEl.textContent = parseInt(countEl.textContent || '0', 10) + 1;
    }

    const title = targetEl.querySelector('h3')?.textContent || 'Target';
    appendLog(`CONFIRMED SELECTION: "${title}"`, 'text-emerald');
  }

  // --- High-Speed Telemetry Polling (40 Hz) & 60-144 FPS Render Loop ---
  let latestState = null;
  let isFetching = false;
  let targetScreenX = window.innerWidth / 2;
  let targetScreenY = window.innerHeight / 2;

  async function pollTelemetry() {
    if (isFetching) return;
    isFetching = true;
    try {
      const resp = await fetch('/api/state');
      if (resp.ok) {
        latestState = await resp.json();
        updateDashboardMetrics(latestState);
      }
    } catch (err) {
      // Reconnecting
    } finally {
      isFetching = false;
      setTimeout(pollTelemetry, 25);
    }
  }

  function updateDashboardMetrics(state) {
    if (!state) return;

    // 1. Top HUD Stats
    if (statLatency) statLatency.textContent = `${state.latency_ms?.toFixed(2) || '0.04'} ms`;
    if (statFps) statFps.textContent = `${state.fps?.toFixed(1) || '30.4'} FPS`;
    if (statConf) statConf.textContent = `${((state.intent_confidence || 0.94) * 100).toFixed(1)}%`;
    if (statCb) statCb.textContent = `${state.execution_provider || 'NPU'} ${state.circuit_breaker_state || 'CLOSED'}`;
    if (statScrollVel) statScrollVel.textContent = `${state.subpixel_velocity !== undefined ? state.subpixel_velocity.toFixed(1) + ' px/s' : '0.0 px/s'}`;
    if (statWatchdog) {
      const isStopped = state.camera_watchdog_status === 'MANUAL_SHUTDOWN';
      statWatchdog.textContent = isStopped ? 'STOPPED' : 'PERMANENT';
      statWatchdog.className = isStopped ? 'stat-value text-red' : 'stat-value text-emerald';
    }

    // 2. Biomarkers
    const ear = state.current_ear || 0.32;
    if (earValue) earValue.textContent = ear.toFixed(3);
    if (earBar) {
      const pct = Math.min(100, Math.max(0, (ear / 0.5) * 100));
      earBar.style.width = `${pct}%`;
      earBar.className = ear < 0.21 ? 'progress-fill fill-cyan' : 'progress-fill fill-cyan';
    }

    // Pupil XY
    const px = state.current_pupil_x || 0.5;
    const py = state.current_pupil_y || 0.5;
    if (pupilCoords) pupilCoords.textContent = `(${px.toFixed(3)}, ${py.toFixed(3)})`;
    if (pupilCrosshair) {
      pupilCrosshair.style.left = `${px * 100}%`;
      pupilCrosshair.style.top = `${py * 100}%`;
    }

    // Head Pose
    const yaw = state.head_yaw ?? 1.2;
    const pitch = state.head_pitch ?? -0.8;
    const roll = state.head_roll ?? 0.4;
    if (headPoseDeg) headPoseDeg.textContent = `Y: ${yaw >= 0 ? '+' : ''}${yaw.toFixed(1)}° P: ${pitch >= 0 ? '+' : ''}${pitch.toFixed(1)}°`;
    if (yawBadge) yawBadge.textContent = `Yaw: ${yaw >= 0 ? '+' : ''}${yaw.toFixed(1)}°`;
    if (pitchBadge) pitchBadge.textContent = `Pitch: ${pitch >= 0 ? '+' : ''}${pitch.toFixed(1)}°`;
    if (rollBadge) rollBadge.textContent = `Roll: ${roll >= 0 ? '+' : ''}${roll.toFixed(1)}°`;

    // Direction & Arrow
    const dir = (state.current_v1_direction || 'center').toUpperCase();
    if (dirText) dirText.textContent = dir;
    if (dirArrow) {
      const arrows = { UP: '⬆️', DOWN: '⬇️', LEFT: '⬅️', RIGHT: '➡️', CENTER: '⏺' };
      dirArrow.textContent = arrows[dir] || '⏺';
    }

    // Directional Scrolling in Sandbox
    if (scrollContent && scrollStatus) {
      if (dir === 'UP') {
        scrollContent.scrollTop -= 6;
        scrollStatus.textContent = 'SCROLLING UP';
        scrollStatus.className = 'badge badge-pulse text-cyan';
      } else if (dir === 'DOWN') {
        scrollContent.scrollTop += 6;
        scrollStatus.textContent = 'SCROLLING DOWN';
        scrollStatus.className = 'badge badge-pulse text-purple';
      } else {
        scrollStatus.textContent = 'RESTING';
        scrollStatus.className = 'badge';
      }
    }
  }

  // --- 60-144 FPS GPU Smooth Render Loop ---
  function renderLoop() {
    if (latestState) {
      const px = latestState.current_pupil_x || 0.5;
      const py = latestState.current_pupil_y || 0.5;
      let rawX = latestState.predicted_screen_x || 0;
      let rawY = latestState.predicted_screen_y || 0;

      if (rawX <= 0 || rawX > window.screen.width) {
        targetScreenX = (1.0 - px) * window.innerWidth;
        targetScreenY = py * window.innerHeight;
      } else {
        targetScreenX = (rawX / window.screen.width) * window.innerWidth;
        targetScreenY = (rawY / window.screen.height) * window.innerHeight;
      }

      // Adaptive dual-speed lerp (responsive saccade snap, stable fixation)
      const dx = targetScreenX - reticleX;
      const dy = targetScreenY - reticleY;
      const dist = Math.hypot(dx, dy);
      const alpha = dist > 90 ? 0.38 : 0.20;
      reticleX += dx * alpha;
      reticleY += dy * alpha;

      if (gazePointer) {
        gazePointer.style.transform = `translate3d(${reticleX - 24}px, ${reticleY - 24}px, 0)`;
      }

      handleGazeHover(reticleX, reticleY, latestState.double_blink_detected);
    }
    requestAnimationFrame(renderLoop);
  }

  function handleGazeHover(x, y, isDoubleBlink) {
    const el = document.elementFromPoint(x, y);
    const targetCard = el ? el.closest('.gaze-target') : null;

    if (targetCard) {
      if (currentTarget !== targetCard) {
        if (currentTarget) currentTarget.classList.remove('gaze-hover');
        currentTarget = targetCard;
        currentTarget.classList.add('gaze-hover');
        dwellStartTime = Date.now();
        if (gazeLabel) gazeLabel.textContent = 'FOCUS TARGET';
        playTone(600, 'sine', 0.04);
      } else {
        // Increment dwell
        const elapsed = Date.now() - dwellStartTime;
        const fraction = Math.min(1.0, elapsed / DWELL_THRESHOLD_MS);
        const offset = 100 - (fraction * 100);
        if (dwellBar) dwellBar.style.strokeDashoffset = offset;

        if (fraction >= 1.0) {
          triggerTargetClick(currentTarget);
          dwellStartTime = Date.now() + 500; // brief cooldown
        }
      }
    } else {
      if (currentTarget) {
        currentTarget.classList.remove('gaze-hover');
        currentTarget = null;
      }
      if (dwellBar) dwellBar.style.strokeDashoffset = 100;
      if (gazeLabel) gazeLabel.textContent = 'LOOKING';
    }

    // Direct double-blink click trigger
    if (isDoubleBlink && currentTarget) {
      triggerTargetClick(currentTarget);
      appendLog('DOUBLE-BLINK INPUT TRIGGER CONFIRMED!', 'text-gold');
    }
  }

  // --- Mode Buttons ---
  modeScrollBtn?.addEventListener('click', () => setOperatingMode('directional_scroll'));
  modeClickBtn?.addEventListener('click', () => setOperatingMode('precision_click'));

  function setOperatingMode(mode) {
    currentMode = mode;
    modeScrollBtn.classList.toggle('active', mode === 'directional_scroll');
    modeClickBtn.classList.toggle('active', mode === 'precision_click');
    playTone(800, 'triangle', 0.08);
    appendLog(`Switched operating mode to: ${mode}`, 'text-purple');
    fetch(`/api/mode?set=${mode}`).catch(() => {});
  }

  // Toggle Overlays
  toggleOverlayBtn?.addEventListener('click', () => {
    showOverlay = !showOverlay;
    toggleOverlayBtn.classList.toggle('active', showOverlay);
    fetch(`/api/overlay?toggle=${showOverlay}`).catch(() => {});
    appendLog(`AR Landmark Overlay: ${showOverlay ? 'ENABLED' : 'DISABLED'}`);
  });

  // Camera Reconnect
  cameraReconnectBtn?.addEventListener('click', () => {
    const streamImg = document.getElementById('camera-stream');
    if (streamImg) {
      streamImg.src = `/video_feed?t=${Date.now()}`;
      appendLog('Webcam video stream connection refreshed.', 'text-gold');
    }
  });

  // Manual Camera Shutdown (the ONLY trigger allowed to terminate camera daemon)
  cameraShutdownBtn?.addEventListener('click', () => {
    if (confirm('Manual Camera Shutdown Policy: Are you sure you want to stop the permanent camera daemon?')) {
      fetch('/api/shutdown')
        .then(r => r.json())
        .then(() => {
          playTone(400, 'sawtooth', 0.2);
          appendLog('MANUAL CAMERA SHUTDOWN EXECUTED BY USER.', 'text-red');
          if (statWatchdog) {
            statWatchdog.textContent = 'STOPPED';
            statWatchdog.className = 'stat-value text-red';
          }
        })
        .catch(() => {});
    }
  });

  // Clear Log
  clearLogBtn?.addEventListener('click', () => {
    if (activityLog) activityLog.innerHTML = '';
  });

  // --- Start App ---
  pollTelemetry();
  requestAnimationFrame(renderLoop);
  appendLog('Connected to FreeSight-OS local streaming server.');
})();
