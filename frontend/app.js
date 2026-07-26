document.addEventListener("DOMContentLoaded", () => {
    const API_BASE = window.location.origin;

    const CITIES = [
        { code: "Delhi", name: "Delhi (DEL)" },
        { code: "Mumbai", name: "Mumbai (BOM)" },
        { code: "Bangalore", name: "Bangalore (BLR)" },
        { code: "Kolkata", name: "Kolkata (CCU)" },
        { code: "Hyderabad", name: "Hyderabad (HYD)" },
        { code: "Chennai", name: "Chennai (MAA)" }
    ];

    // Realistic flight durations by route & stops (hours)
    const ROUTE_DURATIONS = {
        "Delhi-Mumbai": { zero: 2.2, one: 6.5, two_or_more: 12.5 },
        "Delhi-Bangalore": { zero: 2.8, one: 7.5, two_or_more: 14.0 },
        "Delhi-Kolkata": { zero: 2.2, one: 6.0, two_or_more: 11.5 },
        "Delhi-Hyderabad": { zero: 2.2, one: 6.5, two_or_more: 12.0 },
        "Delhi-Chennai": { zero: 2.8, one: 7.5, two_or_more: 13.5 },

        "Mumbai-Delhi": { zero: 2.2, one: 6.5, two_or_more: 12.5 },
        "Mumbai-Bangalore": { zero: 1.8, one: 5.5, two_or_more: 11.0 },
        "Mumbai-Kolkata": { zero: 2.6, one: 7.0, two_or_more: 13.0 },
        "Mumbai-Hyderabad": { zero: 1.5, one: 5.0, two_or_more: 10.0 },
        "Mumbai-Chennai": { zero: 2.0, one: 6.0, two_or_more: 11.5 },

        "Bangalore-Delhi": { zero: 2.8, one: 7.5, two_or_more: 14.0 },
        "Bangalore-Mumbai": { zero: 1.8, one: 5.5, two_or_more: 11.0 },
        "Bangalore-Kolkata": { zero: 2.5, one: 6.8, two_or_more: 12.5 },
        "Bangalore-Hyderabad": { zero: 1.2, one: 4.5, two_or_more: 9.5 },
        "Bangalore-Chennai": { zero: 1.0, one: 4.2, two_or_more: 9.0 },

        "Kolkata-Delhi": { zero: 2.2, one: 6.0, two_or_more: 11.5 },
        "Kolkata-Mumbai": { zero: 2.6, one: 7.0, two_or_more: 13.0 },
        "Kolkata-Bangalore": { zero: 2.5, one: 6.8, two_or_more: 12.5 },
        "Kolkata-Hyderabad": { zero: 2.0, one: 5.8, two_or_more: 11.0 },
        "Kolkata-Chennai": { zero: 2.2, one: 6.2, two_or_more: 11.8 },

        "Hyderabad-Delhi": { zero: 2.2, one: 6.5, two_or_more: 12.0 },
        "Hyderabad-Mumbai": { zero: 1.5, one: 5.0, two_or_more: 10.0 },
        "Hyderabad-Bangalore": { zero: 1.2, one: 4.5, two_or_more: 9.5 },
        "Hyderabad-Kolkata": { zero: 2.0, one: 5.8, two_or_more: 11.0 },
        "Hyderabad-Chennai": { zero: 1.2, one: 4.8, two_or_more: 9.5 },

        "Chennai-Delhi": { zero: 2.8, one: 7.5, two_or_more: 13.5 },
        "Chennai-Mumbai": { zero: 2.0, one: 6.0, two_or_more: 11.5 },
        "Chennai-Bangalore": { zero: 1.0, one: 4.2, two_or_more: 9.0 },
        "Chennai-Kolkata": { zero: 2.2, one: 6.2, two_or_more: 11.8 },
        "Chennai-Hyderabad": { zero: 1.2, one: 4.8, two_or_more: 9.5 },
    };

    let currentCalculatedDuration = 6.5;

    // Tab Navigation Logic
    const navButtons = document.querySelectorAll(".nav-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    function switchTab(targetTabId) {
        navButtons.forEach(btn => {
            if (btn.dataset.tab === targetTabId) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });

        tabPanes.forEach(pane => {
            if (pane.id === targetTabId) {
                pane.classList.add("active");
            } else {
                pane.classList.remove("active");
            }
        });
    }

    navButtons.forEach(btn => {
        btn.addEventListener("click", () => switchTab(btn.dataset.tab));
    });

    document.getElementById("btnGoToShap")?.addEventListener("click", () => switchTab("tab-shap"));

    // Dynamic Destination Filtering & Automatic Route Duration Calculation
    const sourceSelect = document.getElementById("source_city");
    const destSelect = document.getElementById("destination_city");
    const stopsSelect = document.getElementById("stops");
    const durationDisplay = document.getElementById("durationDisplay");
    const btnSwap = document.getElementById("btnSwapCities");

    function updateDestinationOptions() {
        const selectedSource = sourceSelect.value;
        const currentDest = destSelect.value;

        destSelect.innerHTML = "";

        CITIES.forEach(c => {
            if (c.code !== selectedSource) {
                const opt = document.createElement("option");
                opt.value = c.code;
                opt.textContent = c.name;
                if (c.code === currentDest) opt.selected = true;
                destSelect.appendChild(opt);
            }
        });

        if (!destSelect.value && destSelect.options.length > 0) {
            destSelect.options[0].selected = true;
        }

        recalculateRouteDuration();
    }

    function recalculateRouteDuration() {
        const src = sourceSelect.value;
        const dst = destSelect.value;
        const stp = stopsSelect.value;

        const routeKey = `${src}-${dst}`;

        if (ROUTE_DURATIONS[routeKey] && ROUTE_DURATIONS[routeKey][stp]) {
            currentCalculatedDuration = ROUTE_DURATIONS[routeKey][stp];
        } else if (stp === "zero") {
            currentCalculatedDuration = 2.2;
        } else if (stp === "one") {
            currentCalculatedDuration = 6.5;
        } else {
            currentCalculatedDuration = 12.5;
        }

        durationDisplay.textContent = currentCalculatedDuration.toString().replace(".", ",");
    }

    sourceSelect.addEventListener("change", updateDestinationOptions);
    destSelect.addEventListener("change", recalculateRouteDuration);
    stopsSelect.addEventListener("change", recalculateRouteDuration);

    btnSwap?.addEventListener("click", () => {
        const oldSrc = sourceSelect.value;
        const oldDst = destSelect.value;

        sourceSelect.value = oldDst;
        updateDestinationOptions();
        destSelect.value = oldSrc;
        recalculateRouteDuration();
    });

    // Days Left Range Slider
    const daysSlider = document.getElementById("days_left");
    const daysValText = document.getElementById("daysVal");
    daysSlider.addEventListener("input", (e) => {
        daysValText.textContent = e.target.value;
    });

    // Check Backend Health
    async function checkHealth() {
        try {
            const res = await fetch(`${API_BASE}/api/health`);
            if (res.ok) {
                document.getElementById("statusText").textContent = "Backend Conectado";
                document.getElementById("statusIndicator").classList.remove("offline");
            }
        } catch (e) {
            document.getElementById("statusText").textContent = "Backend Inactivo / Reintentando";
        }
    }
    checkHealth();

    // Form Submit (Prediction)
    const predictionForm = document.getElementById("predictionForm");
    predictionForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const payload = {
            airline: document.getElementById("airline").value,
            class: document.getElementById("class_name").value,
            source_city: sourceSelect.value,
            destination_city: destSelect.value,
            departure_time: document.getElementById("departure_time").value,
            arrival_time: document.getElementById("arrival_time").value,
            stops: stopsSelect.value,
            days_left: parseInt(daysSlider.value),
            duration: currentCalculatedDuration, // Automatically computed from route!
        };

        const btnPredict = document.getElementById("btnPredict");
        btnPredict.textContent = "⏳ Calculando Inferencia...";
        btnPredict.disabled = true;

        try {
            // Predict API Call
            const res = await fetch(`${API_BASE}/api/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (res.ok) {
                const data = await res.json();
                document.getElementById("resPriceINR").textContent = data.formatted_price;
                document.getElementById("resPriceUSD").textContent = data.predicted_price_usd.toLocaleString("es-CL");
                document.getElementById("resPriceCLP").textContent = data.predicted_price_clp.toLocaleString("es-CL");
                document.getElementById("resLatency").textContent = `${data.latency_ms} ms`;
            }

            // SHAP Explain API Call
            const explainRes = await fetch(`${API_BASE}/api/explain`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (explainRes.ok) {
                const shapData = await explainRes.json();
                renderShapBars(shapData);
            }

            // Update Telemetry
            updateTelemetry();

        } catch (err) {
            console.error("Error al calcular inferencia:", err);
        } finally {
            btnPredict.textContent = "🚀 Estimar Tarifa del Vuelo";
            btnPredict.disabled = false;
        }
    });

    // Render SHAP Waterfall Bars
    function renderShapBars(shapData) {
        const container = document.getElementById("shapBarsContainer");
        container.innerHTML = "";

        document.getElementById("shapBasePrice").textContent = `₹ ${shapData.base_price_inr.toLocaleString("es-CL")},00`;
        document.getElementById("shapTargetPrice").textContent = `₹ ${shapData.predicted_price_inr.toLocaleString("es-CL")},00`;

        const maxContrib = Math.max(...shapData.contributions.map(c => Math.abs(c.contribution)), 1);

        shapData.contributions.forEach(item => {
            const row = document.createElement("div");
            row.className = "shap-item";

            const percent = Math.min(100, Math.max(10, (Math.abs(item.contribution) / maxContrib) * 100));
            const isIncrease = item.direction === "increases_price";
            const sign = isIncrease ? "+" : "-";

            row.innerHTML = `
                <div class="shap-feature-name">${item.feature}</div>
                <div class="shap-bar-wrapper">
                    <div class="shap-bar-fill ${isIncrease ? 'increase' : 'decrease'}" style="width: ${percent}%;"></div>
                </div>
                <div class="shap-val-text ${isIncrease ? 'increase' : 'decrease'}">
                    ${sign} ₹ ${Math.abs(item.contribution).toLocaleString("es-CL")}
                </div>
            `;

            container.appendChild(row);
        });
    }

    // Telemetry Updater
    async function updateTelemetry() {
        try {
            const res = await fetch(`${API_BASE}/api/metrics`);
            if (res.ok) {
                const metrics = await res.json();
                document.getElementById("telemetryAvgLat").textContent = `${metrics.avg_latency_ms} ms`;
                document.getElementById("telemetryTotalReq").textContent = metrics.total_predictions;
                document.getElementById("telemetryHealth").textContent = metrics.system_health;
            }
        } catch (e) {
            console.warn("Telemetría no disponible:", e);
        }
    }

    // Initialize City Dropdowns on Load
    updateDestinationOptions();
});
