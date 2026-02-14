/* ============================================================
   ADAR V3.0 — Dashboard SocketIO Client
   Connects to the Flask-SocketIO backend and updates
   all HUD elements in real-time.
   ============================================================ */

// ── SocketIO Connection ────────────────────────────────────
const socket = io({ transports: ["websocket", "polling"] });

// ── Chart.js Setup ─────────────────────────────────────────
const GRAPH_MAX_POINTS = 100;
const earData = [];
const marData = [];
const labels = [];
let pointIndex = 0;

const ctx = document.getElementById("earMarChart").getContext("2d");
const earMarChart = new Chart(ctx, {
    type: "line",
    data: {
        labels: labels,
        datasets: [
            {
                label: "EAR",
                data: earData,
                borderColor: "#ffffff",
                backgroundColor: "rgba(255, 255, 255, 0.08)",
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
                fill: true,
            },
            {
                label: "MAR",
                data: marData,
                borderColor: "#ff9100",
                backgroundColor: "rgba(255, 145, 0, 0.1)",
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
                fill: true,
            },
        ],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        interaction: { intersect: false, mode: "index" },
        scales: {
            x: {
                display: false,
            },
            y: {
                min: 0,
                max: 1.2,
                grid: {
                    color: "rgba(255, 140, 0, 0.08)",
                    drawBorder: false,
                },
                ticks: {
                    color: "#4a5568",
                    font: { family: "'Share Tech Mono'", size: 10 },
                    stepSize: 0.2,
                },
            },
        },
        plugins: {
            legend: {
                position: "top",
                labels: {
                    color: "#a0a8b4",
                    font: { family: "'Orbitron'", size: 9 },
                    boxWidth: 20,
                    padding: 15,
                },
            },
            // Threshold annotations drawn manually
        },
    },
    plugins: [
        {
            id: "thresholdLines",
            afterDraw(chart) {
                const yAxis = chart.scales.y;
                const ctx = chart.ctx;

                // EAR threshold line
                const earY = yAxis.getPixelForValue(0.22);
                ctx.save();
                ctx.strokeStyle = "rgba(255, 23, 68, 0.5)";
                ctx.lineWidth = 1;
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.moveTo(chart.chartArea.left, earY);
                ctx.lineTo(chart.chartArea.right, earY);
                ctx.stroke();

                // MAR threshold line
                const marY = yAxis.getPixelForValue(0.75);
                ctx.strokeStyle = "rgba(255, 145, 0, 0.5)";
                ctx.beginPath();
                ctx.moveTo(chart.chartArea.left, marY);
                ctx.lineTo(chart.chartArea.right, marY);
                ctx.stroke();
                ctx.restore();
            },
        },
    ],
});

// ── DOM Elements ───────────────────────────────────────────
const dom = {
    statusBeacon: document.getElementById("statusBeacon"),
    statusLabel: document.getElementById("statusLabel"),
    systemClock: document.getElementById("systemClock"),
    fpsValue: document.getElementById("fpsValue"),
    earValue: document.getElementById("earValue"),
    marValue: document.getElementById("marValue"),
    objectsValue: document.getElementById("objectsValue"),
    headPoseValue: document.getElementById("headPoseValue"),
    drowsinessCard: document.getElementById("drowsinessCard"),
    yawningCard: document.getElementById("yawningCard"),
    distractionCard: document.getElementById("distractionCard"),
    headPoseCard: document.getElementById("headPoseCard"),
    drowsinessIndicator: document.getElementById("drowsinessIndicator"),
    yawningIndicator: document.getElementById("yawningIndicator"),
    distractionIndicator: document.getElementById("distractionIndicator"),
    headPoseIndicator: document.getElementById("headPoseIndicator"),
    jarvisFeed: document.getElementById("jarvisFeed"),
    jarvisDot: document.getElementById("jarvisDot"),
    jarvisStatus: document.getElementById("jarvisStatus"),
    totalAlerts: document.getElementById("totalAlerts"),
    drowsyCount: document.getElementById("drowsyCount"),
    yawnCount: document.getElementById("yawnCount"),
    distractCount: document.getElementById("distractCount"),
    aiLatency: document.getElementById("aiLatency"),
    dangerOverlay: document.getElementById("dangerOverlay"),
    attentionScore: document.getElementById("attentionScore"),
    gaugeArc: document.getElementById("gaugeArc"),
    blinkRate: document.getElementById("blinkRate"),
    faceConf: document.getElementById("faceConf"),
    dangerCount: document.getElementById("dangerCount"),
    sessionUptime: document.getElementById("sessionUptime"),
};

// ── Clock ──────────────────────────────────────────────────
function updateClock() {
    const now = new Date();
    const h = String(now.getHours()).padStart(2, "0");
    const m = String(now.getMinutes()).padStart(2, "0");
    const s = String(now.getSeconds()).padStart(2, "0");
    dom.systemClock.textContent = `${h}:${m}:${s}`;
}
setInterval(updateClock, 1000);
updateClock();

// ── Session Uptime Timer ───────────────────────────────────────
let _sessionStart = Date.now();
function updateSessionUptime() {
    const el = Math.floor((Date.now() - _sessionStart) / 1000);
    const h = String(Math.floor(el / 3600)).padStart(2, "0");
    const m = String(Math.floor((el % 3600) / 60)).padStart(2, "0");
    const s = String(el % 60).padStart(2, "0");
    if (dom.sessionUptime) dom.sessionUptime.textContent = `${h}:${m}:${s}`;
}
setInterval(updateSessionUptime, 1000);

// ── Client-Side Drowsiness Timer (smooth, independent of AI speed) ──
let _drowsyActive = false;
let _drowsyStart = 0;
setInterval(() => {
    if (_drowsyActive) {
        const elapsed = (Date.now() - _drowsyStart) / 1000;
        if (elapsed >= 4.0) {
            dom.earValue.textContent = `🔴 DANGER ${elapsed.toFixed(1)}s`;
        } else {
            dom.earValue.textContent = `⚠️ WARNING ${elapsed.toFixed(1)}s`;
        }
    }
}, 50);  // 20 Hz — buttery smooth

// ── Telemetry Handler ──────────────────────────────────────
socket.on("telemetry_update", (data) => {
    // Update chart
    pointIndex++;
    labels.push(pointIndex);
    earData.push(data.ear);
    marData.push(data.mar);

    if (labels.length > GRAPH_MAX_POINTS) {
        labels.shift();
        earData.shift();
        marData.shift();
    }
    earMarChart.update();

    // Update values
    // Drowsiness timer — driven client-side for smooth updates
    if (data.is_drowsy && !_drowsyActive) {
        _drowsyActive = true;
        _drowsyStart = Date.now();
    } else if (!data.is_drowsy && _drowsyActive) {
        _drowsyActive = false;
        dom.earValue.textContent = `✅ SAFE`;
    }
    // Update drowsy card text with tier
    if (!data.is_drowsy) {
        dom.earValue.textContent = `✅ SAFE`;
    }
    dom.marValue.textContent = `MAR: ${data.mar.toFixed(3)}`;
    dom.headPoseValue.textContent = `YAW: ${data.yaw.toFixed(1)}° | PITCH: ${data.pitch.toFixed(1)}°`;
    dom.fpsValue.textContent = data.camera_fps || "—";
    dom.aiLatency.textContent = `${data.process_time_ms} ms`;

    // Objects
    if (data.detected_objects && data.detected_objects.length > 0) {
        dom.objectsValue.textContent = data.detected_objects.join(", ").toUpperCase();
    } else {
        dom.objectsValue.textContent = "CLEAR";
    }
    
    // Display behavior details if detected
    if (data.behavior_details && data.behavior_details !== "") {
        const behaviorDiv = document.getElementById("behaviorAlert");
        if (behaviorDiv) {
            behaviorDiv.textContent = `⚠️ ${data.behavior_details}`;
            behaviorDiv.style.display = "block";
            behaviorDiv.classList.add("pulse-danger");
        }
    } else {
        const behaviorDiv = document.getElementById("behaviorAlert");
        if (behaviorDiv) {
            behaviorDiv.style.display = "none";
            behaviorDiv.classList.remove("pulse-danger");
        }
    }

    // Update status indicators
    updateIndicator("drowsiness", data.is_drowsy, true);
    updateIndicator("yawning", data.is_yawning, false);
    updateIndicator("distraction", data.is_distracted, true);
    updateIndicator("headPose", data.is_looking_away, false);

    // ── 3-Tier Top Bar: ONLY based on drowsiness ──
    const drowsyDur = data.drowsy_duration || 0;
    const DROWSY_THRESHOLD = 4.0;  // seconds

    if (data.is_drowsy && drowsyDur >= DROWSY_THRESHOLD) {
        // TIER 3: DANGER — drowsy 4s+ → local alert fires
        updateMainStatus("DANGER");
        dom.dangerOverlay.classList.add("active");
    } else if (data.is_drowsy) {
        // TIER 2: WARNING — drowsiness detected, timer running < 4s
        updateMainStatus("WARNING");
        dom.dangerOverlay.classList.remove("active");
    } else {
        // TIER 1: SAFE — no drowsiness detected
        updateMainStatus("SAFE");
        dom.dangerOverlay.classList.remove("active");
    }

    // ── Attention Score Gauge ──
    const score = data.attention_score ?? 100;
    dom.attentionScore.textContent = Math.round(score);
    const circumference = 2 * Math.PI * 85;
    dom.gaugeArc.style.strokeDashoffset = circumference * (1 - score / 100);
    const gaugeColor = score >= 80 ? "var(--accent-green)"
                     : score >= 50 ? "var(--accent-orange)"
                     : "var(--accent-red)";
    dom.gaugeArc.style.stroke = gaugeColor;
    dom.attentionScore.style.color = gaugeColor;

    // Blink rate & face confidence
    dom.blinkRate.textContent = Math.round(data.blink_rate ?? 0);
    const fConf = data.face_confidence ?? 0;
    dom.faceConf.textContent = fConf > 0 ? `${Math.round(fConf * 100)}%` : "—";
    dom.dangerCount.textContent = data.danger_counter ?? 0;
});

function updateIndicator(name, isActive, isDanger) {
    const card = dom[`${name}Card`];
    const indicator = dom[`${name}Indicator`];

    card.classList.remove("active-warning", "active-danger");
    indicator.classList.remove("safe", "warning", "danger");

    if (isActive) {
        if (isDanger) {
            card.classList.add("active-danger");
            indicator.classList.add("danger");
        } else {
            card.classList.add("active-warning");
            indicator.classList.add("warning");
        }
    } else {
        indicator.classList.add("safe");
    }
}

function updateMainStatus(status) {
    const s = status.toLowerCase();
    dom.statusBeacon.className = `status-beacon ${s}`;
    dom.statusLabel.className = `status-label ${s}`;
    dom.statusLabel.textContent = status;
}

// ── Jarvis Alerts ──────────────────────────────────────────
socket.on("jarvis_alert", (data) => {
    addJarvisMessage(data.message || "Alert triggered", "alert");

    // Update stats from server
    fetchStats();
});

socket.on("system_status", (data) => {
    addJarvisMessage(data.message, "system");

    if (data.jarvis_online) {
        dom.jarvisDot.classList.add("active");
        dom.jarvisStatus.textContent = "ONLINE";
    } else {
        dom.jarvisStatus.textContent = "OFFLINE";
    }
});

function addJarvisMessage(text, type = "info") {
    const now = new Date();
    const time = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`;

    const msg = document.createElement("div");
    msg.className = `jarvis-message ${type}`;
    msg.innerHTML = `
        <span class="msg-time">${time}</span>
        <span class="msg-text">${escapeHtml(text)}</span>
    `;

    dom.jarvisFeed.appendChild(msg);
    dom.jarvisFeed.scrollTop = dom.jarvisFeed.scrollHeight;

    // Keep feed clean — max 3 recent messages
    while (dom.jarvisFeed.children.length > 3) {
        dom.jarvisFeed.removeChild(dom.jarvisFeed.firstChild);
    }
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ── Fetch Stats ────────────────────────────────────────────
async function fetchStats() {
    try {
        const res = await fetch("/api/stats");
        const stats = await res.json();
        dom.totalAlerts.textContent = stats.total || 0;
        dom.drowsyCount.textContent = stats.drowsiness || 0;
        dom.yawnCount.textContent = stats.yawning || 0;
        dom.distractCount.textContent = stats.distraction || 0;
    } catch (e) {
        console.warn("Failed to fetch stats:", e);
    }
}

// Refresh stats every 5 seconds
setInterval(fetchStats, 5000);
fetchStats();

// ── Connection Status ──────────────────────────────────────
socket.on("connect", () => {
    console.log("[ADAR] Connected to server");
    _sessionStart = Date.now();
    dom.jarvisFeed.innerHTML = "";
    addJarvisMessage("Systems online — all modules nominal.", "system");
});

socket.on("disconnect", () => {
    console.log("[ADAR] Disconnected");
    updateMainStatus("OFFLINE");
    addJarvisMessage("Connection lost. Attempting reconnect...", "alert");
});

socket.on("connect_error", (err) => {
    console.error("[ADAR] Connection error:", err);
});

console.log(
    "%c ADAR V3.0 — Command Center Loaded ",
    "background: #08090d; color: #ff8c00; font-family: monospace; font-size: 14px; padding: 8px; border: 1px solid #ff8c00;"
);
