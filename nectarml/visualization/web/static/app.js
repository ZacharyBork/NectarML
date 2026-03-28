const graph_panes = {};

function smoothSeries(data, factor) {
    if (factor === 0) return data;
    let last = data[0];
    return data.map(v => {
        last = last * factor + v * (1 - factor);
        return last;
    });
}

function createChartPane(win, title, v_axis_label, h_axis_label) {
    const div = document.createElement("div");
    div.className = "graph-pane";
    div.innerHTML = `<h3>${title || win}</h3><div class="chart-container"><canvas></canvas></div>`;

    const graph_settings = document.createElement("container");
    graph_settings.className = "graph-settings";
    graph_settings.innerHTML = `
    <label for="graph-smoothing">Smoothing:</label>
    <input type="range" id="graph-smoothing" min="0" max="100" value="0">`;

    const slider = graph_settings.querySelector("#graph-smoothing");
    slider.addEventListener("input", () => {
        const factor = slider.value / 101;
        const chart = graph_panes[win].chart;
        const raw = graph_panes[win].raw;
        if (!raw) return;

        chart.data.datasets = raw.Y.map((series, i) => ({
            ...chart.data.datasets[i],
            data: smoothSeries(series, factor).map((y, j) => ({ x: raw.X[j], y }))
        }));
        chart.update();
    });

    const canvas = div.querySelector("canvas");
    let dragStart = null;
    let dragMode = "xy";

    canvas.addEventListener("dblclick", () => {
        graph_panes[win].chart.resetZoom();
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
            scales: {
                x: {
                    type: "linear",
                    ticks: { font: { size: 11 } },
                    title: { display: true, text: h_axis_label, font: { size: 16 } }
                },
                y: {
                    ticks: { font: { size: 11 } },
                    title: { display: true, text: v_axis_label, font: { size: 16 } }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        font: { size: 12 }
                    }
                },
                zoom: {
                    zoom: {
                        drag: { enabled: true },
                        mode: () => dragMode,
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
        win, title,
        X, Y,
        v_axis_label, h_axis_label,
        opts = {}
    } = msg;

    if (!graph_panes[win]) createChartPane(win, title, v_axis_label, h_axis_label);

    const pane = graph_panes[win];
    const chart = pane.chart;
    const factor = pane.el.querySelector("#graph-smoothing").value / 100;

    pane.raw = { X: [...X], Y: Y.map(s => [...s]) };

    chart.data.datasets = Y.map((series, i) => ({
        label: (opts.legend || [])[i] ?? `series ${i}`,
        data: smoothSeries(series, factor).map((y, j) => ({ x: X[j], y })),
        borderWidth: 2,
        pointRadius: 2,
        tension: 0.1,
        fill: false,
    }));

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
