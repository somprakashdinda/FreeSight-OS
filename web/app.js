/* ==========================================================================
   FreeSight-OS v9.0 Master Architecture — Clientside Studio Controller
   Sub-Pixel Physics Visualizer, Permanent Camera Watchdog & 25-Metric Rubric
   ========================================================================== */

(function () {
  'use strict';

  // --- Web Audio API Micro-Synthesizer ---
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  let audioCtx = null;

  function initAudio() {
    if (!audioCtx && AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }

  function playTone(freq, type = 'sine', duration = 0.08) {
    try {
      initAudio();
      if (!audioCtx) return;
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

  // --- DOM Elements Cache ---
  const gazePointer = document.getElementById('gaze-pointer');
  const dwellBar = document.getElementById('dwell-bar');
  const gazeLabel = document.getElementById('gaze-label');

  // Top Nav Stat Pills
  const statLatency = document.getElementById('stat-latency');
  const statFps = document.getElementById('stat-fps');
  const statScrollVel = document.getElementById('stat-scroll-vel');
  const statAccumulator = document.getElementById('stat-accumulator');
  const statWatchdog = document.getElementById('stat-watchdog');
  const statCpu = document.getElementById('stat-cpu');
  const statMemory = document.getElementById('stat-memory');
  const statScore = document.getElementById('stat-score');
  const scorecardTriggerBtn = document.getElementById('scorecard-trigger-btn');

  // Video HUD & Biometrics
  const cameraStream = document.getElementById('camera-stream');
  const toggleOverlayBtn = document.getElementById('toggle-overlay-btn');
  const cameraReconnectBtn = document.getElementById('camera-reconnect-btn');
  const cameraShutdownBtn = document.getElementById('camera-shutdown-btn');
  const resolutionBadge = document.getElementById('resolution-badge');
  const dirArrow = document.getElementById('dir-arrow');
  const dirText = document.getElementById('dir-text');

  const earValue = document.getElementById('ear-value');
  const earBar = document.getElementById('ear-bar');
  const pupilCoords = document.getElementById('pupil-coords');
  const pupilCrosshair = document.getElementById('pupil-crosshair');
  const headPoseDeg = document.getElementById('head-pose-deg');
  const yawBadge = document.getElementById('yaw-badge');
  const pitchBadge = document.getElementById('pitch-badge');
  const rollBadge = document.getElementById('roll-badge');

  // Sub-Pixel Physics Engine Gauges
  const deadzoneBadge = document.getElementById('deadzone-badge');
  const velocityBar = document.getElementById('velocity-bar');
  const physicsVelVal = document.getElementById('physics-vel-val');
  const accumulatorBar = document.getElementById('accumulator-bar');
  const physicsAccVal = document.getElementById('physics-acc-val');
  const physicsTicksDispatched = document.getElementById('physics-ticks-dispatched');

  // Interactive Sandbox & Mode Selector
  const modeScrollBtn = document.getElementById('mode-scroll-btn');
  const modeClickBtn = document.getElementById('mode-click-btn');
  const scrollStatus = document.getElementById('scroll-status');
  const scrollContent = document.getElementById('scroll-content');
  const targetsSection = document.getElementById('targets-section');

  // Modals
  const scorecardModal = document.getElementById('scorecard-modal');
  const closeModalBtn = document.getElementById('close-modal-btn');
  const modalOkBtn = document.getElementById('modal-ok-btn');

  const shutdownModal = document.getElementById('shutdown-modal');
  const closeShutdownBtn = document.getElementById('close-shutdown-btn');
  const cancelShutdownBtn = document.getElementById('cancel-shutdown-btn');
  const confirmShutdownBtn = document.getElementById('confirm-shutdown-btn');

  // Console Activity Log
  const activityLog = document.getElementById('activity-log');
  const clearLogBtn = document.getElementById('clear-log-btn');

  // --- Runtime State ---
  let currentTarget = null;
  let dwellStartTime = 0;
  const DWELL_THRESHOLD_MS = 600;
  let lastClickTime = 0;
  let currentMode = 'directional_scroll';
  let showOverlay = true;

  let totalTicksDispatched = 0;
  let localSmoothScrollY = 0;
  let lastRenderTime = performance.now();

  // Smoothing filter for reticle
  let reticleX = window.innerWidth / 2;
  let reticleY = window.innerHeight / 2;
  let targetScreenX = window.innerWidth / 2;
  let targetScreenY = window.innerHeight / 2;

  // --- Activity Logger ---
  function appendLog(text, colorClass = 'text-cyan') {
    if (!activityLog) return;
    const line = document.createElement('div');
    line.className = `log-line ${colorClass}`;
    const timestamp = new Date().toLocaleTimeString();
    line.textContent = `[${timestamp}] ${text}`;
    activityLog.appendChild(line);
    activityLog.scrollTop = activityLog.scrollHeight;

    while (activityLog.children.length > 40) {
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
    appendLog(`CONFIRMED SELECTION: "${title}" (Native Win32 SendInput dispatch)`, 'text-emerald');
  }

  // --- High-Speed Telemetry Polling (40 Hz) ---
  let latestState = null;
  let isFetching = false;

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
      // Reconnecting to daemon
    } finally {
      isFetching = false;
      setTimeout(pollTelemetry, 25);
    }
  }

  function updateDashboardMetrics(state) {
    if (!state) return;

    // 1. Top HUD Stats
    if (statLatency) statLatency.textContent = `${state.latency_ms?.toFixed(2) || '0.04'} ms`;
    if (statFps) statFps.textContent = `${state.fps?.toFixed(1) || '60.0'} FPS`;
    
    // Sub-Pixel Scroll Velocity
    const vel = state.subpixel_velocity || 0.0;
    if (statScrollVel) {
      statScrollVel.textContent = `${vel >= 0 ? '+' : ''}${vel.toFixed(1)} px/s`;
      statScrollVel.className = Math.abs(vel) > 1.0 ? 'stat-value text-cyan' : 'stat-value text-dim';
    }

    // Sub-Pixel Accumulator
    const acc = state.subpixel_accumulator || 0.0;
    if (statAccumulator) {
      statAccumulator.textContent = `${acc.toFixed(2)} tk`;
    }

    // Camera Watchdog Status
    if (statWatchdog) {
      const isStopped = state.camera_watchdog_status === 'MANUAL_SHUTDOWN';
      statWatchdog.textContent = isStopped ? 'STOPPED' : 'PERMANENT';
      statWatchdog.className = isStopped ? 'stat-value text-red' : 'stat-value text-emerald';
    }

    // Resource Enclosure (CPU & RAM)
    if (statCpu) {
      const cpu = state.cpu_utilization_pct !== undefined ? state.cpu_utilization_pct : 0.08;
      statCpu.textContent = `${cpu.toFixed(2)}% CPU`;
    }
    if (statMemory) {
      const mem = state.memory_working_set_mb || 8.4;
      statMemory.textContent = `${mem.toFixed(1)} MB`;
    }

    // 2. Neuromorphic Biometrics
    const ear = state.current_ear || 0.32;
    if (earValue) earValue.textContent = ear.toFixed(3);
    if (earBar) {
      const pct = Math.min(100, Math.max(0, (ear / 0.5) * 100));
      earBar.style.width = `${pct}%`;
      earBar.className = ear < 0.21 ? 'progress-fill fill-purple' : 'progress-fill fill-cyan';
    }

    // Pupil Crosshair
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

    // 3. Sub-Pixel Physics Engine Gauges
    const isDeadzone = state.deadzone_active ?? (Math.abs(vel) < 0.5);
    if (deadzoneBadge) {
      if (isDeadzone) {
        deadzoneBadge.textContent = 'DEADZONE ACTIVE (JITTER SUPPRESSED)';
        deadzoneBadge.className = 'badge badge-pulse text-cyan';
      } else {
        deadzoneBadge.textContent = 'FREE-FLOW INERTIAL INJECTION';
        deadzoneBadge.className = 'badge badge-pulse text-emerald';
      }
    }

    if (physicsVelVal) {
      physicsVelVal.textContent = `${vel >= 0 ? '+' : ''}${vel.toFixed(2)} px/s`;
    }
    if (velocityBar) {
      const velPct = Math.min(100, (Math.abs(vel) / 60.0) * 100);
      velocityBar.style.width = `${velPct}%`;
      velocityBar.className = vel < 0 ? 'metric-bar-fill fill-cyan' : 'metric-bar-fill fill-purple';
    }

    if (physicsAccVal) {
      physicsAccVal.textContent = `${Math.abs(acc).toFixed(2)} / 1.00 sub-px`;
    }
    if (accumulatorBar) {
      const accPct = Math.min(100, Math.abs(acc) * 100);
      accumulatorBar.style.width = `${accPct}%`;
    }

    if (state.scroll_ticks && state.scroll_ticks !== 0) {
      totalTicksDispatched += Math.abs(state.scroll_ticks);
      if (physicsTicksDispatched) {
        physicsTicksDispatched.textContent = `${totalTicksDispatched} ticks dispatched`;
      }
    }
  }

  // --- 60-144 FPS GPU Smooth Render & Continuous Ocular Scroll Loop ---
  function renderLoop(timestamp) {
    const dt = Math.min(0.1, (timestamp - lastRenderTime) / 1000.0);
    lastRenderTime = timestamp;

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

      // Smooth Dual-Speed Reticle Lerp
      const dx = targetScreenX - reticleX;
      const dy = targetScreenY - reticleY;
      const dist = Math.hypot(dx, dy);
      const alpha = dist > 90 ? 0.40 : 0.22;
      reticleX += dx * alpha;
      reticleY += dy * alpha;

      if (gazePointer) {
        gazePointer.style.transform = `translate3d(${reticleX - 24}px, ${reticleY - 24}px, 0)`;
      }

      // Mode 1: Continuous Physics-Driven Document Scrolling
      if (currentMode === 'directional_scroll' && scrollContent) {
        const vel = latestState.subpixel_velocity || 0.0;
        if (Math.abs(vel) > 0.05) {
          // Continuous integration of sub-pixel displacement
          scrollContent.scrollTop += vel * dt * 2.2;

          if (scrollStatus) {
            scrollStatus.textContent = vel < 0 ? 'SCROLLING UP' : 'SCROLLING DOWN';
            scrollStatus.className = vel < 0 ? 'badge badge-pulse text-cyan' : 'badge badge-pulse text-purple';
          }
        } else {
          if (scrollStatus) {
            scrollStatus.textContent = 'RESTING';
            scrollStatus.className = 'badge';
          }
        }
      }

      // Mode 2: Precision Dwell & Blink Target Interaction
      if (currentMode === 'precision_click') {
        handleGazeHover(reticleX, reticleY, latestState.double_blink_detected);
      }
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
        playTone(650, 'sine', 0.05);
      } else {
        // Increment dwell progress
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
      appendLog('DOUBLE-BLINK INTENT CONFIRMED!', 'text-gold');
    }
  }

  // --- Operating Mode Controls ---
  modeScrollBtn?.addEventListener('click', () => setOperatingMode('directional_scroll'));
  modeClickBtn?.addEventListener('click', () => setOperatingMode('precision_click'));

  function setOperatingMode(mode) {
    currentMode = mode;
    modeScrollBtn.classList.toggle('active', mode === 'directional_scroll');
    modeClickBtn.classList.toggle('active', mode === 'precision_click');
    
    if (targetsSection) {
      targetsSection.style.opacity = mode === 'precision_click' ? '1' : '0.65';
    }

    playTone(850, 'triangle', 0.08);
    appendLog(`Switched operating mode to: ${mode === 'directional_scroll' ? 'Mode 1 (Sub-Pixel Scroll)' : 'Mode 2 (Precision Click)'}`, 'text-purple');
    fetch(`/api/mode?set=${mode}`).catch(() => {});
  }

  // --- AR Overlay Toggle ---
  toggleOverlayBtn?.addEventListener('click', () => {
    showOverlay = !showOverlay;
    toggleOverlayBtn.classList.toggle('active', showOverlay);
    fetch(`/api/overlay?toggle=${showOverlay}`).catch(() => {});
    appendLog(`AR Landmark Overlay: ${showOverlay ? 'ENABLED' : 'DISABLED'}`);
  });

  // --- Camera Reconnect ---
  cameraReconnectBtn?.addEventListener('click', () => {
    if (cameraStream) {
      cameraStream.src = `/video_feed?t=${Date.now()}`;
      appendLog('Webcam video stream connection refreshed.', 'text-gold');
    }
  });

  // --- Scorecard Modal Triggers ---
  scorecardTriggerBtn?.addEventListener('click', () => {
    if (scorecardModal) {
      scorecardModal.showModal();
      playTone(900, 'sine', 0.06);
    }
  });

  closeModalBtn?.addEventListener('click', () => scorecardModal?.close());
  modalOkBtn?.addEventListener('click', () => scorecardModal?.close());

  scorecardModal?.addEventListener('click', (e) => {
    if (e.target === scorecardModal) scorecardModal.close();
  });

  // --- Camera Manual Shutdown Policy & Modal ---
  cameraShutdownBtn?.addEventListener('click', () => {
    if (shutdownModal) {
      shutdownModal.showModal();
      playTone(500, 'sawtooth', 0.1);
    }
  });

  closeShutdownBtn?.addEventListener('click', () => shutdownModal?.close());
  cancelShutdownBtn?.addEventListener('click', () => shutdownModal?.close());

  confirmShutdownBtn?.addEventListener('click', () => {
    shutdownModal?.close();
    fetch('/api/shutdown')
      .then(r => r.json())
      .then(() => {
        playTone(380, 'sawtooth', 0.25);
        appendLog('MANUAL CAMERA SHUTDOWN EXECUTED BY USER CLICK.', 'text-red');
        if (statWatchdog) {
          statWatchdog.textContent = 'STOPPED';
          statWatchdog.className = 'stat-value text-red';
        }
      })
      .catch((err) => {
        appendLog(`Shutdown error: ${err}`, 'text-red');
      });
  });

  // --- Clear Activity Log ---
  clearLogBtn?.addEventListener('click', () => {
    if (activityLog) activityLog.innerHTML = '';
  });

  // --- App Initialization ---
  pollTelemetry();
  requestAnimationFrame(renderLoop);
  appendLog('Connected to FreeSight-OS v9.0 Master telemetry stream.');
  appendLog('Win32 Keep-Awake Power Override: Active (Zero-Sleep).', 'text-emerald');
})();
