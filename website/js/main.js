'use strict';

/* ── Nav scroll effect ──────────────────────────────────────────────────── */
const nav = document.getElementById('nav');
window.addEventListener('scroll', () => {
  nav.classList.toggle('scrolled', window.scrollY > 60);
}, { passive: true });

/* ── Animated stat counters ─────────────────────────────────────────────── */
function animateCounter(el, target, decimals = 0, duration = 1800) {
  const start = performance.now();
  const step = (now) => {
    const p = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    el.textContent = (target * ease).toFixed(decimals);
    if (p < 1) requestAnimationFrame(step);
    else el.textContent = target.toFixed(decimals);
  };
  requestAnimationFrame(step);
}

const statObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (!e.isIntersecting) return;
    const el = e.target;
    const target = parseFloat(el.dataset.target);
    const decimals = (el.dataset.target.includes('.')) ? 1 : 0;
    animateCounter(el, target, decimals);
    statObserver.unobserve(el);
  });
}, { threshold: 0.5 });

document.querySelectorAll('.stat-val[data-target]').forEach(el => statObserver.observe(el));

/* ── Timeline bar animation ─────────────────────────────────────────────── */
const tlObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.querySelectorAll('.tl-bar').forEach((bar, i) => {
      const finalWidth = bar.style.width;
      bar.style.width = '0';
      setTimeout(() => { bar.style.width = finalWidth; }, i * 120);
    });
    tlObserver.unobserve(e.target);
  });
}, { threshold: 0.3 });

const tl = document.querySelector('.outbreak-timeline');
if (tl) {
  tl.querySelectorAll('.tl-bar').forEach(b => b.style.transition = 'width 0.8s cubic-bezier(0.2,1,0.3,1)');
  tlObserver.observe(tl);
}

/* ── Heatmap canvas background ──────────────────────────────────────────── */
const canvas = document.getElementById('heatmap-canvas');
if (canvas) {
  const ctx = canvas.getContext('2d');
  let W, H, nodes, animFrame;

  // Zimbabwe district approximate positions (x%, y%) on canvas
  const DISTRICTS = [
    { x: 0.55, y: 0.25, r: 0.8, name: 'Harare' },
    { x: 0.30, y: 0.72, r: 0.6, name: 'Bulawayo' },
    { x: 0.58, y: 0.30, r: 0.5, name: 'Chitungwiza' },
    { x: 0.80, y: 0.38, r: 0.4, name: 'Mutare' },
    { x: 0.45, y: 0.52, r: 0.35, name: 'Gweru' },
    { x: 0.50, y: 0.42, r: 0.3, name: 'Kwekwe' },
    { x: 0.42, y: 0.38, r: 0.25, name: 'Kadoma' },
    { x: 0.58, y: 0.62, r: 0.35, name: 'Masvingo' },
    { x: 0.43, y: 0.25, r: 0.25, name: 'Chinhoyi' },
    { x: 0.65, y: 0.28, r: 0.2, name: 'Marondera' },
    { x: 0.52, y: 0.68, r: 0.2, name: 'Zvishavane' },
    { x: 0.40, y: 0.30, r: 0.2, name: 'Chegutu' },
    { x: 0.60, y: 0.20, r: 0.2, name: 'Bindura' },
    { x: 0.48, y: 0.90, r: 0.3, name: 'Beitbridge' },
    { x: 0.18, y: 0.42, r: 0.25, name: 'Hwange' },
  ];

  const RISK_COLORS = [
    { r: 16,  g: 185, b: 129 },  // green (low)
    { r: 245, g: 158, b: 11  },  // yellow (medium)
    { r: 232, g: 64,  b: 64  },  // red (high)
  ];

  function resize() {
    W = canvas.width  = canvas.offsetWidth  * devicePixelRatio;
    H = canvas.height = canvas.offsetHeight * devicePixelRatio;
  }

  function drawHeatmap(t) {
    ctx.clearRect(0, 0, W, H);
    DISTRICTS.forEach((d, i) => {
      const cx = d.x * W;
      const cy = d.y * H;
      // Animate intensity with slow sine wave per district
      const intensity = 0.3 + 0.7 * (0.5 + 0.5 * Math.sin(t * 0.0004 + i * 1.2));
      const riskIdx = Math.floor(intensity * 2.99);
      const rc = RISK_COLORS[riskIdx];
      const radius = d.r * Math.min(W, H) * 0.12 * (0.8 + 0.2 * intensity);

      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
      grad.addColorStop(0,   `rgba(${rc.r},${rc.g},${rc.b},${0.18 * intensity})`);
      grad.addColorStop(0.5, `rgba(${rc.r},${rc.g},${rc.b},${0.08 * intensity})`);
      grad.addColorStop(1,   `rgba(${rc.r},${rc.g},${rc.b},0)`);

      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
    });
  }

  function loop(t) {
    drawHeatmap(t);
    animFrame = requestAnimationFrame(loop);
  }

  resize();
  window.addEventListener('resize', resize, { passive: true });
  requestAnimationFrame(loop);
}

/* ── API code tab switcher ──────────────────────────────────────────────── */
window.showTab = function(id) {
  document.querySelectorAll('.code-block').forEach(b => b.style.display = 'none');
  document.querySelectorAll('.code-tab').forEach(t => t.classList.remove('active'));
  const block = document.getElementById('tab-' + id);
  if (block) block.style.display = 'block';
  event.target.classList.add('active');
};

/* ── Feature cards fade-in on scroll ───────────────────────────────────── */
const fadeObserver = new IntersectionObserver((entries) => {
  entries.forEach((e, i) => {
    if (!e.isIntersecting) return;
    setTimeout(() => {
      e.target.style.opacity = '1';
      e.target.style.transform = 'translateY(0)';
    }, i * 60);
    fadeObserver.unobserve(e.target);
  });
}, { threshold: 0.1 });

document.querySelectorAll('.feature-card, .sec-item, .api-ep, .team-card').forEach(el => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(20px)';
  el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
  fadeObserver.observe(el);
});

/* ── Smooth anchor links ────────────────────────────────────────────────── */
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const target = document.querySelector(a.getAttribute('href'));
    if (!target) return;
    e.preventDefault();
    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
});

console.log('%cCholSurv Zimbabwe — Open Source Health Intelligence', 'color:#e84040;font-family:monospace;font-size:14px;font-weight:bold');
console.log('%cGitHub: https://github.com/YOUR_USERNAME/cholera-zim-forecast', 'color:#8494a8;font-family:monospace;font-size:12px');
