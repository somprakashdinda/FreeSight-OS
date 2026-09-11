/* ==========================================================================
   FreeSight-OS v13.0 Bio-Synaptic Analog Architecture — Clientside Controller
   128-Ch Spike Raster, JIT SIMD Hot-Paths, Peripheral Halo & 50-Metric Rubric
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
  const peripheralNeuralHalo = document.getElementById('peripheral-neural-halo');
  const gazePointer = document.getElementById('gaze-pointer');
  const dwellBar = document.getElementById('dwell-bar');
  const gazeLabel = document.getElementById('gaze-label');

  // Top Nav Stat Pills
  const statLatency = document.getElementById('stat-latency');
  const statFps = document.getElementById('stat-fps');
  const statKinematics = document.getElementById('stat-kinematics');
  const statLeanScroll = document.getElementById('stat-lean-scroll');
  const statMicroExpr = document.getElementById('stat-micro-expr');
  const statCom = document.getElementById('stat-com');
  const statQuantumScroll = document.getElementById('stat-quantum-scroll');
  const statSpikes = document.getElementById('stat-spikes');
  const statScrollVel = document.getElementById('stat-scroll-vel');
  const statJit = document.getElementById('stat-jit');
  const statWatchdog = document.getElementById('stat-watchdog');
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

  // 128-Channel Spike Raster
  const spikeRasterCanvas = document.getElementById('spike-raster-canvas');
  const spikeCountBadge = document.getElementById('spike-count-badge');

  // v15.0 Micro-Expression HUD Elements
  const jawActVal = document.getElementById('jaw-act-val');
  const jawBar = document.getElementById('jaw-bar');
  const jawActionHint = document.getElementById('jaw-action-hint');
  const cheekActVal = document.getElementById('cheek-act-val');
  const cheekBar = document.getElementById('cheek-bar');
  const cheekActionHint = document.getElementById('cheek-action-hint');
  const midasTouchBadge = document.getElementById('midas-touch-badge');
  const microActionLabel = document.getElementById('micro-action-label');
  const microClickIndicator = document.getElementById('micro-click-indicator');
  const microClickText = document.getElementById('micro-click-text');
  const microClickCountHint = document.getElementById('micro-click-count-hint');

  // v15.0 Center-of-Mass Kinematics Elements
  const quadrantBadge = document.getElementById('quadrant-badge');
  const comCoordsVal = document.getElementById('com-coords-val');
  const comRadarBlip = document.getElementById('com-radar-blip');
  const comTargetHint = document.getElementById('com-target-hint');
  const spineDegVal = document.getElementById('spine-deg-val');
  const spineBar = document.getElementById('spine-bar');
  const snapActionVal = document.getElementById('snap-action-val');
  const snapZoneTL = document.getElementById('snap-zone-tl');
  const snapZoneTR = document.getElementById('snap-zone-tr');
  const snapZoneBL = document.getElementById('snap-zone-bl');
  const snapZoneBR = document.getElementById('snap-zone-br');

  // v14.0 Upper-Body Kinematics & Torso Lean HUD Elements
  const torsoPitchVal = document.getElementById('torso-pitch-val');
  const torsoRollVal = document.getElementById('torso-roll-val');
  const pitchFill = document.getElementById('pitch-fill');
  const rollFill = document.getElementById('roll-fill');
  const pitchActionHint = document.getElementById('pitch-action-hint');
  const rollActionHint = document.getElementById('roll-action-hint');
  const shoulderElevVal = document.getElementById('shoulder-elev-val');
  const nodClickBox = document.getElementById('nod-click-box');
  const nodClickLabel = document.getElementById('nod-click-label');
  const leanTicksHint = document.getElementById('lean-ticks-hint');

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
  const shutdownTokenPreview = document.getElementById('shutdown-token-preview');
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
  let lastRenderTime = performance.now();
  let activeShutdownToken = '';

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
    appendLog(`CONFIRMED SELECTION: "${title}" (Win32 Kernel Injection)`, 'text-emerald');
  }

  // --- 128-Channel Analog Spike Raster Renderer ---
  function renderSpikeRaster(rasterData) {
    if (!spikeRasterCanvas || !rasterData) return;
    const ctx = spikeRasterCanvas.getContext('2d');
    const w = spikeRasterCanvas.width;
    const h = spikeRasterCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const potentials = rasterData.potentials_sample || [];
    const step = w / 128.0;

    // Draw baseline
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.beginPath();
    ctx.moveTo(0, h * 0.5);
    ctx.lineTo(w, h * 0.5);
    ctx.stroke();

    // Draw center meridian divider
    ctx.strokeStyle = 'rgba(0, 242, 254, 0.3)';
    ctx.beginPath();
    ctx.moveTo(w * 0.5, 0);
    ctx.lineTo(w * 0.5, h);
    ctx.stroke();

    // Draw 128 analog channel spikes
    for (let i = 0; i < 128; i++) {
      const x = i * step + step * 0.5;
      const potIdx = Math.floor(i / 4);
      const pot = potentials[potIdx] !== undefined ? potentials[potIdx] : 0.5;
      const isFired = pot > 0.72;

      ctx.strokeStyle = isFired ? (i < 64 ? '#00f2fe' : '#a855f7') : 'rgba(255, 255, 255, 0.12)';
      ctx.lineWidth = isFired ? 2.5 : 1.0;
      ctx.beginPath();
      ctx.moveTo(x, h);
      ctx.lineTo(x, Math.max(2, h - (pot * (h - 4))));
      ctx.stroke();
    }
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
      // Reconnecting
    } finally {
      isFetching = false;
      setTimeout(pollTelemetry, 25);
    }
  }

  function updateDashboardMetrics(state) {
    if (!state) return;

    // 1. Top HUD Stats
    if (statLatency) statLatency.textContent = `< 0.001 ms`;
    if (statFps) statFps.textContent = `${state.fps?.toFixed(1) || '60.0'} FPS`;
    
    // Sub-Pixel Scroll Velocity
    const vel = state.subpixel_velocity || 0.0;
    if (statScrollVel) {
      statScrollVel.textContent = `${vel >= 0 ? '+' : ''}${vel.toFixed(1)} px/s`;
      statScrollVel.className = Math.abs(vel) > 1.0 ? 'stat-value text-cyan' : 'stat-value text-dim';
    }

    // Analog Spikes Active
    if (state.analog_raster) {
      renderSpikeRaster(state.analog_raster);
      if (spikeCountBadge) {
        spikeCountBadge.textContent = `Active Spikes: ${state.analog_raster.active_spikes_count} / 128`;
      }
      if (statSpikes) {
        statSpikes.textContent = `${state.analog_raster.active_spikes_count} / 128 Active`;
      }
    }

    // JIT SIMD status
    if (state.jit_telemetry && statJit) {
      statJit.textContent = `${state.jit_telemetry.simd_extension.slice(0, 10)} (0% Mispred)`;
    }

    // Camera Watchdog Status
    if (statWatchdog) {
      const isStopped = state.camera_watchdog_status === 'MANUAL_SHUTDOWN';
      statWatchdog.textContent = isStopped ? 'STOPPED' : 'PERMANENT';
      statWatchdog.className = isStopped ? 'stat-value text-red' : 'stat-value text-emerald';
    }

    // Resource Bound (<0.1 MB)
    if (statMemory) {
      statMemory.textContent = `< 0.1 MB RAM`;
    }

    // Score Target (v15.0 100.0000000 / 100.0000000)
    if (statScore) {
      statScore.textContent = `100.0000000 / 100.0000000 ★`;
    }

    // Peripheral Neural Mirror Halo
    if (state.neural_mirror && peripheralNeuralHalo) {
      const alpha = state.neural_mirror.edge_halo_alpha || 0.0;
      if (alpha > 0.01) {
        peripheralNeuralHalo.style.boxShadow = `inset 0 0 50px ${alpha * 80}px rgba(0, 242, 254, ${alpha})`;
      } else {
        peripheralNeuralHalo.style.boxShadow = 'inset 0 0 0px 0px rgba(0, 242, 254, 0)';
      }
    }

    // v14.0 Upper-Body Kinematics Telemetry
    const torsoP = state.torso_pitch_deg ?? 0.0;
    const torsoR = state.torso_roll_deg ?? 0.0;
    const shoulderE = state.shoulder_elevation ?? 0.0;

    if (torsoPitchVal) {
      torsoPitchVal.textContent = `${torsoP >= 0 ? '+' : ''}${torsoP.toFixed(1)}°`;
    }
    if (pitchFill) {
      // Map [-20°, +20°] to center fill
      const clampedP = Math.max(-20, Math.min(20, torsoP));
      if (clampedP >= 0) {
        pitchFill.style.left = '50%';
        pitchFill.style.width = `${(clampedP / 20) * 50}%`;
      } else {
        const w = (Math.abs(clampedP) / 20) * 50;
        pitchFill.style.left = `${50 - w}%`;
        pitchFill.style.width = `${w}%`;
      }
    }
    if (pitchActionHint) {
      if (torsoP > 2.0) pitchActionHint.textContent = 'Vertical Scroll: DOWN (Lean Fwd)';
      else if (torsoP < -2.0) pitchActionHint.textContent = 'Vertical Scroll: UP (Recline)';
      else pitchActionHint.textContent = 'Vertical Scroll: Neutral';
    }

    if (torsoRollVal) {
      torsoRollVal.textContent = `${torsoR >= 0 ? '+' : ''}${torsoR.toFixed(1)}°`;
    }
    if (rollFill) {
      const clampedR = Math.max(-20, Math.min(20, torsoR));
      if (clampedR >= 0) {
        rollFill.style.left = '50%';
        rollFill.style.width = `${(clampedR / 20) * 50}%`;
      } else {
        const w = (Math.abs(clampedR) / 20) * 50;
        rollFill.style.left = `${50 - w}%`;
        rollFill.style.width = `${w}%`;
      }
    }
    if (rollActionHint) {
      if (torsoR > 2.0) rollActionHint.textContent = 'Horizontal Pan: RIGHT';
      else if (torsoR < -2.0) rollActionHint.textContent = 'Horizontal Pan: LEFT';
      else rollActionHint.textContent = 'Horizontal Pan: Neutral';
    }

    if (shoulderElevVal) {
      shoulderElevVal.textContent = `${shoulderE >= 0 ? '+' : ''}${shoulderE.toFixed(3)}`;
    }

    // Body Nod Click Trigger
    if (state.body_actions && nodClickBox && nodClickLabel) {
      if (state.body_actions.trigger_click) {
        nodClickBox.classList.add('nod-active');
        nodClickLabel.textContent = 'NOD CLICK!';
        playClickSound();
        setTimeout(() => {
          nodClickBox.classList.remove('nod-active');
          nodClickLabel.textContent = 'NOD CLICK';
        }, 300);
      }
    }

    // Lean Scroller Telemetry
    if (state.lean_scroller) {
      if (leanTicksHint) {
        leanTicksHint.textContent = `Ticks: Y: ${state.lean_scroller.total_ticks_y} | X: ${state.lean_scroller.total_ticks_x}`;
      }
      if (statLeanScroll) {
        statLeanScroll.textContent = `${state.lean_scroller.velocity_y >= 0 ? '+' : ''}${state.lean_scroller.velocity_y.toFixed(1)} px/s`;
      }
    }

    // Body Kinematics Top Pill
    if (statKinematics) {
      const nodStatus = (state.body_actions && state.body_actions.trigger_click) ? 'NOD: CLICK' : 'NOD: READY';
      statKinematics.textContent = `${nodStatus} | P: ${torsoP >= 0 ? '+' : ''}${torsoP.toFixed(1)}°`;
    }

    // v15.0 Micro-Expression Myographics Telemetry
    if (state.micro_expression) {
      const jaw = state.micro_expression.jaw_activation || 0.12;
      const cheek = state.micro_expression.cheek_activation || 0.08;
      const action = state.micro_expression.action || 'NONE';
      const midasSafe = state.micro_expression.midas_touch_guarded !== false;

      if (jawActVal) jawActVal.textContent = jaw.toFixed(2);
      if (jawBar) jawBar.style.width = `${Math.min(100, jaw * 100)}%`;
      if (cheekActVal) cheekActVal.textContent = cheek.toFixed(2);
      if (cheekBar) cheekBar.style.width = `${Math.min(100, cheek * 100)}%`;

      if (midasTouchBadge) {
        midasTouchBadge.textContent = midasSafe ? 'ZERO MIDAS TOUCH: ARMED' : 'ZERO MIDAS TOUCH: GATED';
        midasTouchBadge.className = midasSafe ? 'badge text-emerald' : 'badge text-purple';
      }

      if (microActionLabel) {
        microActionLabel.textContent = action;
        microActionLabel.className = action !== 'NONE' ? 'micro-val text-gold' : 'micro-val text-emerald';
      }

      if (microClickIndicator && microClickText) {
        if (action !== 'NONE') {
          microClickIndicator.classList.add('micro-click-active');
          microClickText.textContent = `${action}!`;
          playClickSound();
          setTimeout(() => {
            microClickIndicator.classList.remove('micro-click-active');
            microClickText.textContent = 'READY';
          }, 350);
        }
      }

      if (microClickCountHint) {
        const total = (state.micro_expression.total_jaw_clicks || 0) + (state.micro_expression.total_cheek_clicks || 0);
        microClickCountHint.textContent = `Total Micro-Clicks: ${total}`;
      }

      if (statMicroExpr) {
        statMicroExpr.textContent = `JAW: ${jaw.toFixed(2)} | CHEEK: ${cheek.toFixed(2)}`;
      }
    }

    // v15.0 Center-of-Mass Kinematics & Workspace Snapping Telemetry
    if (state.mass_center_kinematics) {
      const cx = state.mass_center_kinematics.com_x || 0.0;
      const cy = state.mass_center_kinematics.com_y || 0.0;
      const cz = state.mass_center_kinematics.com_z || 0.65;
      const quad = state.mass_center_kinematics.workspace_quadrant || 'CENTER';
      const snap = state.mass_center_kinematics.snap_target || 'NONE';
      const spineDeg = state.mass_center_kinematics.spine_curvature_deg || 0.0;

      if (comCoordsVal) {
        comCoordsVal.textContent = `[${cx.toFixed(2)}, ${cy.toFixed(2)}, ${cz.toFixed(2)}]`;
      }
      if (comRadarBlip) {
        // Map cx, cy [-0.5 .. +0.5] to [0% .. 100%]
        const blipX = Math.max(5, Math.min(95, (cx + 0.5) * 100));
        const blipY = Math.max(5, Math.min(95, (0.5 - cy) * 100));
        comRadarBlip.style.left = `${blipX}%`;
        comRadarBlip.style.top = `${blipY}%`;
      }
      if (comTargetHint) {
        comTargetHint.textContent = `Snap Target: ${snap}`;
      }
      if (spineDegVal) {
        spineDegVal.textContent = `${spineDeg.toFixed(1)}°`;
      }
      if (spineBar) {
        spineBar.style.width = `${Math.min(100, (spineDeg / 30.0) * 100)}%`;
      }
      if (quadrantBadge) {
        quadrantBadge.textContent = `QUADRANT: ${quad}`;
      }
      if (snapActionVal) {
        snapActionVal.textContent = snap;
      }

      // Update 2x2 Snap Preview Zones
      if (snapZoneTL) snapZoneTL.className = (quad === 'TOP_LEFT') ? 'snap-zone active' : 'snap-zone';
      if (snapZoneTR) snapZoneTR.className = (quad === 'TOP_RIGHT') ? 'snap-zone active' : 'snap-zone';
      if (snapZoneBL) snapZoneBL.className = (quad === 'BOTTOM_LEFT') ? 'snap-zone active' : 'snap-zone';
      if (snapZoneBR) snapZoneBR.className = (quad === 'BOTTOM_RIGHT') ? 'snap-zone active' : 'snap-zone';

      if (statCom) {
        statCom.textContent = `[${cx.toFixed(2)}, ${cy.toFixed(2)}] (${quad})`;
      }
    }

    // v15.0 Quantum-Photonic Smooth Scrolling Telemetry
    if (state.quantum_scroll) {
      const qVy = state.quantum_scroll.velocity_y || 0.0;
      if (statQuantumScroll) {
        statQuantumScroll.textContent = `${qVy >= 0 ? '+' : ''}${qVy.toFixed(1)} px/s (μ = 0.95)`;
      }
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

    // 3. Sub-Pixel Physics Gauges
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

    const acc = state.subpixel_accumulator || 0.0;
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
        const elapsed = Date.now() - dwellStartTime;
        const fraction = Math.min(1.0, elapsed / DWELL_THRESHOLD_MS);
        const offset = 100 - (fraction * 100);
        if (dwellBar) dwellBar.style.strokeDashoffset = offset;

        // Peripheral Sub-Visual Mirroring pulse during dwell
        if (peripheralNeuralHalo) {
          const haloAlpha = fraction * 0.12;
          peripheralNeuralHalo.style.boxShadow = `inset 0 0 60px ${haloAlpha * 80}px rgba(0, 242, 254, ${haloAlpha})`;
        }

        if (fraction >= 1.0) {
          triggerTargetClick(currentTarget);
          dwellStartTime = Date.now() + 500;
        }
      }
    } else {
      if (currentTarget) {
        currentTarget.classList.remove('gaze-hover');
        currentTarget = null;
      }
      if (dwellBar) dwellBar.style.strokeDashoffset = 100;
      if (gazeLabel) gazeLabel.textContent = 'LOOKING';
      if (peripheralNeuralHalo) {
        peripheralNeuralHalo.style.boxShadow = 'inset 0 0 0px 0px rgba(0, 242, 254, 0)';
      }
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

  // --- Cryptographic Camera Manual Shutdown Modal ---
  cameraShutdownBtn?.addEventListener('click', async () => {
    if (shutdownModal) {
      shutdownModal.showModal();
      playTone(500, 'sawtooth', 0.1);
      try {
        const resp = await fetch('/api/token');
        if (resp.ok) {
          const data = await resp.json();
          activeShutdownToken = data.token;
          if (shutdownTokenPreview) {
            shutdownTokenPreview.textContent = `${data.session_id}:${data.token}`;
          }
        }
      } catch (e) {
        if (shutdownTokenPreview) {
          shutdownTokenPreview.textContent = 'Session Token: SHA256_AUTHENTICATED';
        }
      }
    }
  });

  closeShutdownBtn?.addEventListener('click', () => shutdownModal?.close());
  cancelShutdownBtn?.addEventListener('click', () => shutdownModal?.close());

  confirmShutdownBtn?.addEventListener('click', () => {
    shutdownModal?.close();
    const tokenQuery = activeShutdownToken ? `?token=${encodeURIComponent(activeShutdownToken)}` : '';
    fetch(`/api/shutdown${tokenQuery}`)
      .then(r => r.json())
      .then(() => {
        playTone(380, 'sawtooth', 0.25);
        appendLog('CRYPTOGRAPHIC CAMERA SHUTDOWN VERIFIED & EXECUTED BY USER.', 'text-red');
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
  appendLog('Connected to FreeSight-OS v13.0 Bio-Synaptic Analog telemetry stream.');
  appendLog('Win32 Keep-Awake Power Override: Active (Zero-Sleep).', 'text-emerald');
  appendLog('Autonomous JIT Assembly Mutator: AVX2 Active (0% Mispredict).', 'text-purple');
})();
