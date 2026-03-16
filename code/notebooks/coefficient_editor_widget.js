function cloneWeights(weights) {
  return (weights || []).map((row) => [...row]);
}

function formatScaleValue(value) {
  if (Number.isInteger(value)) {
    return String(value);
  }
  return Number(value).toFixed(2).replace(/\.?0+$/, "");
}

function render({ model, el }) {
  let isDragging = false;

  const updateUI = () => {
    const title = model.get("title") || "Coefficient Editor";
    const subtitle = model.get("subtitle") || "";
    const features = model.get("features") || [];
    const channelLabels = model.get("channel_labels") || [];
    const weights = model.get("weights") || [];
    const sliderMin = model.get("slider_min");
    const sliderMax = model.get("slider_max");
    const sliderStep = model.get("slider_step");
    const sliderMid =
      sliderMin <= 0 && sliderMax >= 0 ? 0 : (sliderMin + sliderMax) / 2;

    if (!features.length || !weights.length) {
      el.innerHTML = `
        <div class="ce-shell">
          <div class="ce-empty">No editable coefficients for the selected state.</div>
        </div>
      `;
      return;
    }

    const legend = channelLabels
      .map(
        (label, idx) => `
          <span class="ce-legend-item">
            <span class="ce-legend-dot ce-dot-${idx % 3}"></span>
            ${label}
          </span>
        `,
      )
      .join("");

    const channels = features
      .map((feature, featureIdx) => {
        const sliders = weights
          .map((row, channelIdx) => {
            const value = Number(row[featureIdx] ?? 0);
            return `
              <div class="ce-slider-col">
                <div class="ce-value ce-value-${channelIdx % 3}">${value.toFixed(2)}</div>
                <div class="ce-rail">
                  <input
                    type="range"
                    class="ce-slider ce-slider-${channelIdx % 3}"
                    orient="vertical"
                    min="${sliderMin}"
                    max="${sliderMax}"
                    step="${sliderStep}"
                    value="${value}"
                    data-feature-idx="${featureIdx}"
                    data-channel-idx="${channelIdx}"
                    aria-label="${feature} ${channelLabels[channelIdx] || `Channel ${channelIdx + 1}`}"
                  />
                </div>
                <div class="ce-channel-label">${channelLabels[channelIdx] || `C${channelIdx + 1}`}</div>
              </div>
            `;
          })
          .join("");

        return `
          <div class="ce-channel-card">
            <div class="ce-channel-header">${feature}</div>
            <div class="ce-slider-stack">
              <div class="ce-scale">
                <span class="ce-scale-mark">${formatScaleValue(sliderMax)}</span>
                <span class="ce-scale-mark">${formatScaleValue(sliderMid)}</span>
                <span class="ce-scale-mark">${formatScaleValue(sliderMin)}</span>
              </div>
              ${sliders}
            </div>
          </div>
        `;
      })
      .join("");

    el.innerHTML = `
      <div class="ce-shell">
        <div class="ce-header">
          <div class="ce-heading">
            <div class="ce-title">${title}</div>
            <div class="ce-subtitle">${subtitle}</div>
          </div>
          <div class="ce-actions">
            <div class="ce-legend">${legend}</div>
            <button class="ce-reset" type="button">Reset</button>
          </div>
        </div>
        <div class="ce-board">${channels}</div>
      </div>
    `;

    const sliders = el.querySelectorAll(".ce-slider");
    sliders.forEach((slider) => {
      slider.addEventListener("pointerdown", () => {
        isDragging = true;
      });
      slider.addEventListener("input", (event) => {
        const featureIdx = Number(event.target.dataset.featureIdx);
        const channelIdx = Number(event.target.dataset.channelIdx);
        const value = Number(event.target.value);
        const next = cloneWeights(model.get("weights"));
        next[channelIdx][featureIdx] = value;
        model.set("weights", next);
        const valueNode = event.target
          .closest(".ce-slider-col")
          .querySelector(".ce-value");
        if (valueNode) {
          valueNode.textContent = value.toFixed(2);
        }
        model.save_changes();
      });
    });

    const resetButton = el.querySelector(".ce-reset");
    if (resetButton) {
      resetButton.addEventListener("click", () => {
        model.set("weights", cloneWeights(model.get("original_weights")));
        model.save_changes();
      });
    }
  };

  const stopDragging = () => {
    if (!isDragging) {
      return;
    }
    isDragging = false;
    updateUI();
  };

  updateUI();
  document.addEventListener("pointerup", stopDragging);
  model.on("change:title", updateUI);
  model.on("change:subtitle", updateUI);
  model.on("change:features", updateUI);
  model.on("change:channel_labels", updateUI);
  model.on("change:original_weights", updateUI);
  model.on("change:weights", () => {
    if (!isDragging) {
      updateUI();
    }
  });
}

export default { render };
