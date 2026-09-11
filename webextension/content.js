/**
 * FreeSight DOPC Enterprise - Content Script
 * Injects sub-millisecond ocular reticle and dispatches kinematic scrolling and clicks.
 */

(function () {
  if (window.__DOPC_INJECTED__) return;
  window.__DOPC_INJECTED__ = true;

  // Create reticle element
  const reticle = document.createElement("div");
  reticle.id = "dopc-ocular-reticle";
  reticle.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    width: 28px;
    height: 28px;
    margin-left: -14px;
    margin-top: -14px;
    border-radius: 50%;
    border: 2px solid #06b6d4;
    background: rgba(6, 182, 212, 0.15);
    box-shadow: 0 0 12px rgba(6, 182, 212, 0.6), inset 0 0 6px rgba(99, 102, 241, 0.5);
    pointer-events: none;
    z-index: 2147483647;
    transition: transform 0.05s linear, border-color 0.15s ease, background 0.15s ease;
    transform: translate3d(960px, 540px, 0);
  `;

  // Center crosshair dot
  const dot = document.createElement("div");
  dot.style.cssText = `
    position: absolute;
    top: 50%;
    left: 50%;
    width: 4px;
    height: 4px;
    margin-left: -2px;
    margin-top: -2px;
    border-radius: 50%;
    background: #6366f1;
    box-shadow: 0 0 6px #6366f1;
  `;
  reticle.appendChild(dot);
  document.documentElement.appendChild(reticle);

  let currentX = window.innerWidth / 2;
  let currentY = window.innerHeight / 2;

  // Listen for telemetry from background service worker
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type !== "DOPC_TELEMETRY" || !msg.data) return;
    const d = msg.data;

    // Scale to window viewport
    const screenWidth = window.screen.width || 1920;
    const screenHeight = window.screen.height || 1080;
    const clientX = (d.gaze_x / screenWidth) * window.innerWidth;
    const clientY = (d.gaze_y / screenHeight) * window.innerHeight;

    currentX = clientX;
    currentY = clientY;
    reticle.style.transform = `translate3d(${clientX}px, ${clientY}px, 0)`;

    // Handle chest dip left click
    if (d.click_event || d.chest_dip_click) {
      reticle.style.borderColor = "#22c55e";
      reticle.style.background = "rgba(34, 197, 94, 0.4)";
      setTimeout(() => {
        reticle.style.borderColor = "#06b6d4";
        reticle.style.background = "rgba(6, 182, 212, 0.15)";
      }, 150);

      // Trigger click event on DOM target
      const target = document.elementFromPoint(clientX, clientY);
      if (target) {
        target.click();
      }
    }

    // Handle torso kinetic scrolling
    if (d.torso_scroll_velocity && Math.abs(d.torso_scroll_velocity) > 0.5) {
      window.scrollBy({
        top: d.torso_scroll_velocity * 0.25,
        behavior: "auto"
      });
    }
  });

  console.log("[DOPC Enterprise] In-browser ocular precision reticle active.");
})();
