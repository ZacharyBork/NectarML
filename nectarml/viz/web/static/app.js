const graph_panes = {};

const COLORS = [
    "rgb(252, 50, 0)",
    "rgb(70, 230, 90)",
    "rgb(50, 90, 237)",
    "rgb(252, 129, 74)",
    "rgb(154, 230, 180)",
    "rgb(99, 179, 237)",
    "rgb(183, 148, 246)",
    "rgb(246, 173, 85)",
];

function toFaded(rgb, alpha) {
    return rgb.replace("rgb(", "rgba(").replace(")", `, ${alpha})`);
}

function smoothSeries(data, factor) {
    if (factor === 0) return data;
    let last = data[0];
    return data.map(v => {
        last = last * factor + v * (1 - factor);
        return last;
    });
}

function applySmoothing(chart, raw, factor, colors, labels) {
    const datasets = [];

    raw.Y.forEach((series, i) => {
        const label = labels[i];
        const color = colors[i];

        if (factor > 0) {
            datasets.push({
                label: label,
                data: series.map((y, j) => ({ x: raw.X[j], y })),
                borderWidth: 1,
                pointRadius: 1,
                tension: 0.1,
                borderColor: toFaded(color, 0.25),
            });
            datasets.push({
                label: `${label} (smoothed)`,
                data: smoothSeries(series, factor).map((y, j) => ({ x: raw.X[j], y })),
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.1,
                borderColor: color,
            });
        } else {
            datasets.push({
                label: label,
                data: series.map((y, j) => ({ x: raw.X[j], y })),
                borderWidth: 2,
                pointRadius: 1,
                tension: 0.1,
                borderColor: color,
            });
        }
    });

    chart.data.datasets = datasets;
    chart.update();
}


function createChartPane(win, title, v_axis_label, h_axis_label) {
    const div = document.createElement("div");
    div.className = "graph-pane";
    div.innerHTML = `
        <h3>${title || win}</h3>
        <div class="chart-container">
            <canvas></canvas>
            <div class="zoom-hint">Double click to reset zoom</div>
        </div>`;

    const graph_settings = document.createElement("container");
    graph_settings.className = "graph-settings";
    graph_settings.innerHTML = `
    <label for="graph-smoothing">Smoothing:</label>
    <input type="range" id="graph-smoothing" min="0" max="95" value="0">`;

    const slider = graph_settings.querySelector("#graph-smoothing");
    slider.addEventListener("input", () => {
        const factor = slider.value / 100;
        const pane = graph_panes[win];
        if (!pane.raw) return;
        applySmoothing(pane.chart, pane.raw, factor, pane.colors, pane.labels);
    });

    const canvas = div.querySelector("canvas");
    let dragStart = null;
    let dragMode = "xy";

    canvas.addEventListener("dblclick", () => {
        graph_panes[win].chart.resetZoom();
        div.querySelector(".zoom-hint").style.display = "none";
    });

    canvas.addEventListener("mousedown", (e) => {
        dragStart = { x: e.clientX, y: e.clientY };
        dragMode = "xy";
    });

    canvas.addEventListener("mousemove", (e) => {
        if (!dragStart) return;
        const dx = Math.abs(e.clientX - dragStart.x);
        const dy = Math.abs(e.clientY - dragStart.y);
        if (dx > 10 || dy > 10) {
            if (dx > dy * 40) dragMode = "x";
            else if (dy > dx * 2) dragMode = "y";
            else dragMode = "xy";
        }
    });

    canvas.addEventListener("mouseup", () => {
        dragStart = null;
    });

    div.appendChild(graph_settings);

    document.getElementById("graph-container").appendChild(div);

    graph_panes[win] = { el: div, chart: null };

    const chart = new Chart(div.querySelector("canvas"), {
        type: "line",
        data: { labels: [], datasets: [] },
        options: {
            animation: false,
            responsive: true,
            maintainAspectRatio: false,
            devicePixelRatio: window.devicePixelRatio,
            fill: false,
            scales: {
                x: {
                    type: "linear",
                    ticks: { font: { family: "'JetBrains Mono', monospace", size: 11 } },
                    title: { display: true, text: h_axis_label, font: { size: 16 } }
                },
                y: {
                    ticks: { font: { family: "'JetBrains Mono', monospace", size: 11 } },
                    title: { display: true, text: v_axis_label, font: { size: 16 } }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        font: { 
                            family: "'JetBrains Mono', monospace",
                            size: 10 
                        }
                    }
                },
                zoom: {
                    zoom: {
                        drag: { enabled: true },
                        mode: () => dragMode,
                        onZoomComplete() {
                            div.querySelector(".zoom-hint").style.display = "block";
                        }
                    },
                    limits: {
                        x: { min: "original", max: "original" },
                        y: { min: "original", max: "original" }
                    }
                }
            }
        }
    });

    graph_panes[win].chart = chart;
}

function updateLine(msg) {
    const {
        win, title, X, Y,
        legend, v_axis_label, h_axis_label,
        opts = {}
    } = msg;

    if (!graph_panes[win]) createChartPane(win, title, v_axis_label, h_axis_label);

    const pane = graph_panes[win];
    const chart = pane.chart;
    const factor = pane.el.querySelector("#graph-smoothing").value / 100;

    if (!pane.colors) {
        pane.colors = Y.map((_, i) => COLORS[i % COLORS.length]);
    }
    if (!pane.labels) {
        pane.labels = Y.map((_, i) => (legend || [])[i] ?? `series ${i}`);
    }

    pane.raw = { X: [...X], Y: Y.map(s => [...s]) };
    if (!pane.colors) {
        pane.colors = Y.map((_, i) => COLORS[i % COLORS.length]);
    }
    applySmoothing(chart, pane.raw, factor, pane.colors, pane.labels);

    if (opts.xlabel) chart.options.scales.x.title.text = opts.xlabel;
    if (opts.ylabel) chart.options.scales.y.title.text = opts.ylabel;

    chart.update();
}

const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    if (msg.type === "line" || msg.type === "line_update") {
        updateLine(msg);
    }
};

const image_panes = {};

function createImagePane(win, title) {
    const div = document.createElement("div");
    div.className = "image-pane";
    div.innerHTML = `<h3>${title || win}</h3><img></img>`;
    document.getElementById("image-container").appendChild(div);
    image_panes[win] = { el: div, chart: null };
}

function updateImage(msg) {
    const { win, title, data } = msg;

    if (!image_panes[win]) createImagePane(win, title);

    image_panes[win].el.querySelector("img").src = `data:image/png;base64,${data}`;
}

ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    if (msg.type === "line" || msg.type === "line_update") {
        updateLine(msg);
    } else if (msg.type === "image") {
        updateImage(msg);
    }
};

function clearAll() {
    document.getElementById("graph-container").innerHTML = "";
    for (const win in graph_panes) delete graph_panes[win];
    document.getElementById("image-container").innerHTML = "";
    for (const win in image_panes) delete image_panes[win];
}

ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    if (msg.type === "line" || msg.type === "line_update") {
        updateLine(msg);
    } else if (msg.type === "image") {
        updateImage(msg);
    } else if (msg.type === "clear") {
        clearAll();
    }
};

const clear_button = document.getElementById("ClearButton");
if (clear_button) {
    clear_button.addEventListener("click", clearAll);
}
