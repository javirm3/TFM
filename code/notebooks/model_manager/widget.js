/**
 * ModelManagerWidget — anywidget render function.
 *
 * Python is the source of truth for all data (groups, options, defaults).
 * This file only handles rendering and wiring DOM events back to traitlets.
 */

// ── Rendering helpers ─────────────────────────────────────────────────────────

/**
 * Render a grouped regressor selector table.
 *
 * @param {Array}  groups      - emission_groups or transition_groups traitlet value
 * @param {Array}  selectedCols - emission_cols or transition_cols traitlet value
 * @param {string} dataAttr    - "emission" or "transition"
 *
 * Table columns: Label | L | N/C | R
 * - Row label cell: clicking toggles all members in that row
 * - Individual cells: clicking toggles just that regressor
 * - Row gets "selected" (all members active) or "partial" (some active) class
 */
function renderRegressorTable(groups, selectedCols, dataAttr) {
  if (!groups || groups.length === 0) {
    return '<p class="mm-empty-note">No regressors available.</p>';
  }

  const sel = new Set(selectedCols);

  let rows = "";
  for (const group of groups) {
    const members   = group.members;           // { L?, C?, N?, R? } → col name
    const allCols   = Object.values(members);
    const allSel    = allCols.every(c => sel.has(c));
    const someSel   = allCols.some(c =>  sel.has(c));
    const rowClass  = allSel ? "selected" : someSel ? "partial" : "";

    // Encode member list in the label cell so the click handler can toggle all
    const membersJSON = JSON.stringify(allCols).replace(/'/g, "&#39;");

    const cellFor = (side) => {
      const col = members[side];
      if (!col) return `<td class="mm-reg-cell mm-reg-empty"></td>`;
      const active = sel.has(col) ? "selected" : "";
      return `<td class="mm-reg-cell ${active}" data-${dataAttr}="${col}">${col}</td>`;
    };

    // N/C column: prefer C (sided neutral) over N (global)
    const ncCell = members["C"] ? cellFor("C") : cellFor("N");

    rows += `
      <tr class="mm-reg-row ${rowClass}">
        <td class="mm-reg-row-label ${rowClass}"
            data-${dataAttr}-group="${group.key}"
            data-${dataAttr}-members='${membersJSON}'
        >${group.label}</td>
        ${cellFor("L")}${ncCell}${cellFor("R")}
      </tr>`;
  }

  return `
    <table class="mm-reg-table">
      <thead>
        <tr>
          <th class="mm-reg-th-label">Regressor</th>
          <th class="mm-reg-th-side">L</th>
          <th class="mm-reg-th-side">N / C</th>
          <th class="mm-reg-th-side">R</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

/**
 * Render the "Select All" toggle button above the subjects grid.
 * State classes: "selected" (all), "partial" (some), "" (none).
 */
function renderSelectAll(subjectsList, currentSubjects) {
  const allSel  = subjectsList.length > 0 && currentSubjects.length === subjectsList.length;
  const someSel = currentSubjects.length > 0 && currentSubjects.length < subjectsList.length;
  const cls     = allSel ? "selected" : someSel ? "partial" : "";
  const label   = allSel ? "Deselect All" : "Select All";
  return `<button class="mm-select-all ${cls}" id="btn-select-all">${label}</button>`;
}

function renderFrozenTable(numStates, selectedFeatures, frozenEmissions) {
  if (!selectedFeatures || selectedFeatures.length === 0) {
    return '<p class="mm-empty-note">Select emission regressors to freeze per state.</p>';
  }

  const frozen = frozenEmissions || {};
  const head = selectedFeatures
    .map(
      (feature) => `
        <th class="mm-freeze-th" title="${feature}">
          <span class="mm-freeze-head-text">${feature}</span>
        </th>
      `
    )
    .join("");

  const rows = Array.from({ length: numStates }, (_, stateIdx) => {
    const stateKey = String(stateIdx);
    const stateFrozen = frozen[stateKey] || {};
    const cells = selectedFeatures
      .map((feature) => {
        const value = stateFrozen[feature];
        const display = Number.isFinite(value) ? String(value) : "";
        return `
          <td class="mm-freeze-td">
            <input
              type="number"
              step="any"
              class="mm-freeze-input"
              data-state="${stateKey}"
              data-feature="${feature}"
              value="${display}"
              placeholder="free"
            >
          </td>
        `;
      })
      .join("");

    return `
      <tr class="mm-freeze-row">
        <td class="mm-freeze-state">state ${stateIdx}</td>
        ${cells}
      </tr>
    `;
  }).join("");

  return `
    <div class="mm-freeze-table-wrap">
      <table class="mm-freeze-table">
        <thead>
          <tr>
            <th class="mm-freeze-th-state">State</th>
            ${head}
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

// ── Main render ───────────────────────────────────────────────────────────────

function render({ model, el }) {
  const containerId = "mm-" + Math.random().toString(36).substring(7);
  let aliasDraft = model.get("alias") || "";
  let aliasDirty = false;

  const updateUI = () => {
    const existingVal      = model.get("existing_model");
    const is2afc           = model.get("is_2afc");
    const isRunning        = model.get("is_running");
    const modelType        = model.get("model_type");
    const currentTask      = model.get("task");
    const taskOptions      = model.get("task_options") || [];
    const mode             = model.get("ui_mode");
    const KList            = model.get("k_options");
    const currentK         = model.get("K");
    const subjectsList     = model.get("subjects_list");
    const currentSubjects  = model.get("subjects");
    const currentEmission  = model.get("emission_cols");
    const currentTransition = model.get("transition_cols");
    const currentFrozen    = model.get("frozen_emissions") || {};
    const currentTau       = model.get("tau");
    const currentLapse     = model.get("lapse");
    const currentLapseMax  = model.get("lapse_max");
    const currentAlias     = model.get("alias");
    const aliasError       = model.get("alias_error");
    const aliasStatus      = model.get("alias_status");
    const savedModelName   = model.get("saved_model_name");
    const existingInfo     = model.get("existing_models_info");
    const emissionGroups   = model.get("emission_groups");
    const transitionGroups = model.get("transition_groups");

    if (!aliasDirty) {
      aliasDraft = currentAlias;
    }

    const aliasMessage = aliasDirty
      ? ""
      : (aliasError || aliasStatus || (savedModelName ? `Current saved model: ${savedModelName}` : ""));
    const aliasMessageClass = aliasError
      ? "error"
      : (aliasStatus ? "status" : "hint");

    // ── Shell ───────────────────────────────────────────────────────────────
    let html = `
      <div class="mm-container" id="${containerId}">
        <div class="mm-header">
            <div class="mm-task-selector">
              <label class="mm-label inline">Task:</label>
              <select id="inp-task" class="mm-input small">
                ${taskOptions.map(opt => `
                  <option value="${opt.value}" ${currentTask === opt.value ? "selected" : ""}>${opt.label}</option>
                `).join("")}
              </select>
            </div>
          </div>
        <div class="mm-tabs">
          <button class="mm-tab ${mode === 'new'  ? 'active' : ''}" data-mode="new">New Fit</button>
          <button class="mm-tab ${mode === 'load' ? 'active' : ''}" data-mode="load">Load Existing</button>
        </div>
        <div class="mm-content">
    `;

    // ── Load Existing tab ───────────────────────────────────────────────────
    if (mode === "load") {
      const showTransitionRegressors = modelType === "glmhmmt";
      html += `
        <div class="mm-section">
          <label class="mm-label">Select Saved Model</label>
          <div class="mm-table-container">
            <table class="mm-table">
              <thead>
                <tr>
                  <th>Model Name</th>
                  <th>Subjects</th>
                  <th>K</th>
                  <th>Regressors</th>
                  ${showTransitionRegressors ? "<th>Transition Regressors</th>" : ""}
                  <th>Tau</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                ${existingInfo.map(info => {
                  const isDefault  = info.id === "__default__";
                  const isSelected = info.id === existingVal;
                  return `
                    <tr class="mm-tr ${isSelected ? "selected" : ""} ${isDefault ? "mm-tr-default" : ""}"
                        data-model="${info.id}">
                      <td>
                        <strong>${info.name}</strong>
                        ${isDefault ? '<span class="mm-default-badge">default</span>' : ""}
                      </td>
                      <td>${info.subjects}</td>
                      <td>${info.K}</td>
                      <td class="mm-wrap">${info.regressors}</td>
                      ${showTransitionRegressors ? `<td class="mm-wrap">${info.transition_regressors || ""}</td>` : ""}
                      <td>${info.tau}</td>
                      <td class="mm-actions-cell">
                        ${isDefault ? "" : `<button class="mm-btn-delete-row" data-delete-model="${info.id}">Delete</button>`}
                      </td>
                    </tr>`;
                }).join("")}
              </tbody>
            </table>
          </div>
        </div>
      `;

    // ── New Fit tab ─────────────────────────────────────────────────────────
    } else {
      html += `
        <div class="mm-flex-row">
          <div class="mm-col">
            <div class="mm-section">
              <label class="mm-label">Subjects</label>
              ${renderSelectAll(subjectsList, currentSubjects)}
              <div class="mm-chip-container subjects-grid">
                ${subjectsList.map(s => `
                  <div class="mm-chip ${currentSubjects.includes(s) ? "selected" : ""}"
                       data-subject="${s}">${s}</div>
                `).join("")}
              </div>
            </div>
      `;

      if (modelType !== "glm") {
        html += `
            <div class="mm-section">
              <label class="mm-label">Number of States (K)</label>
              <div class="mm-slider-wrap">
                <input type="range" class="mm-range" id="inp-k-range"
                       min="${Math.min(...KList)}" max="${Math.max(...KList)}"
                       value="${currentK}" step="1">
                <input type="number" class="mm-num-input" id="inp-k-num"
                       min="${Math.min(...KList)}" max="${Math.max(...KList)}"
                       value="${currentK}" step="1">
              </div>
            </div>
        `;
      }

      html += `
          </div>

          <div class="mm-col">
            <div class="mm-section">
              <label class="mm-label">Emission Regressors</label>
              ${renderRegressorTable(emissionGroups, currentEmission, "emission")}
            </div>
      `;

      if (modelType === "glmhmmt") {
        html += `
            <div class="mm-section">
              <label class="mm-label">Transition Regressors</label>
              ${renderRegressorTable(transitionGroups, currentTransition, "transition")}
            </div>
        `;
      }

      html += `
          </div>
        </div>
      `;

      if (modelType !== "glm") {
        html += `
        <div class="mm-section">
          <label class="mm-label">Frozen Emission Weights</label>
          ${renderFrozenTable(currentK, currentEmission, currentFrozen)}
        </div>
        `;
      }

      html += `
        <hr class="mm-divider"/>

        <div class="mm-flex-row">
          <div class="mm-col half-col">
            <div class="mm-section">
              <label class="mm-label">Action Trace Half-life (τ)</label>
              <div class="mm-slider-wrap">
                <input type="range" class="mm-range" id="inp-tau-range"
                       min="1" max="200" value="${currentTau}" step="1">
                <input type="number" class="mm-num-input" id="inp-tau-num"
                       min="1" max="200" value="${currentTau}" step="1">
              </div>
            </div>
          </div>
      `;

      if (modelType === "glm" && is2afc) {
        html += `
          <div class="mm-col">
            <div class="mm-section row-align">
              <label class="mm-checkbox">
                <input type="checkbox" id="inp-lapse" ${currentLapse ? "checked" : ""}>
                <span class="mm-checkmark"></span>
                Fit Lapse Rates
              </label>
              <div class="mm-slider-wrap tight ${!currentLapse ? "disabled" : ""}">
                <span class="mm-label inline">Max Lapse:</span>
                <input type="range" class="mm-range" id="inp-lapse-max-range"
                       min="0.01" max="0.5" value="${currentLapseMax}" step="0.01"
                       ${!currentLapse ? "disabled" : ""}>
                <input type="number" class="mm-num-input" id="inp-lapse-max-num"
                       min="0.01" max="0.5" value="${currentLapseMax}" step="0.01"
                       ${!currentLapse ? "disabled" : ""}>
              </div>
            </div>
          </div>
        `;
      }

      html += `</div>`;  // end flex-row (tau/lapse)
    } // end new fit block

    // ── Footer (always visible) ─────────────────────────────────────────────
    html += `
        <hr class="mm-divider"/>
        <div class="mm-footer">
          <div class="mm-alias-wrap">
            <label class="mm-label inline">Custom Alias:</label>
            <div class="mm-alias-controls">
              <input type="text" id="inp-alias" class="mm-input ${(!aliasDirty && aliasError) ? "error" : ""}"
                     placeholder="e.g. my_best_fit">
              <button class="mm-btn-secondary" id="btn-save-alias" ${isRunning ? "disabled" : ""}>Save</button>
            </div>
            <div class="mm-alias-message ${aliasMessageClass}">${aliasMessage}</div>
          </div>
          <button class="mm-btn-run" id="btn-run" ${isRunning ? "disabled" : ""}>
            ${isRunning ? "FITTING..." : "RUN FIT"}
          </button>
        </div>
      </div>
    </div>
    `;

    el.innerHTML = html;
    const aliasInput = el.querySelector("#inp-alias");
    if (aliasInput) {
      aliasInput.value = aliasDraft;
    }

    // ── Event wiring helpers ────────────────────────────────────────────────
    const bind    = (sel, ev, fn) => { const n = el.querySelector(sel);    if (n) n.addEventListener(ev, fn); };
    const bindAll = (sel, ev, fn) => { el.querySelectorAll(sel).forEach(n => n.addEventListener(ev, fn)); };

    // Tabs
    bindAll(".mm-tab", "click", (e) => {
      model.set("ui_mode", e.target.dataset.mode);
      model.save_changes();
    });

    // Load table row click — sets existing_model; Python observer does the heavy lifting
    bindAll(".mm-tr", "click", (e) => {
      if (e.target.closest(".mm-btn-delete-row")) return;
      const row = e.target.closest(".mm-tr");
      if (!row) return;
      el.querySelectorAll(".mm-tr").forEach(r => r.classList.remove("selected"));
      row.classList.add("selected");
      model.set("existing_model", row.dataset.model);
      model.save_changes();
    });

    bindAll(".mm-btn-delete-row", "click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const name = e.target.dataset.deleteModel;
      if (!name) return;
      if (!window.confirm(`Delete model "${name}" and its folder?`)) return;
      model.set("delete_model_name", name);
      model.set("delete_model_clicks", model.get("delete_model_clicks") + 1);
      model.save_changes();
    });

    // Select All toggle
    bind("#btn-select-all", "click", () => {
      const list = model.get("subjects_list");
      const cur  = model.get("subjects");
      model.set("subjects", cur.length === list.length ? [] : [...list]);
      model.save_changes();
    });

    // Subject chips
    bindAll(".mm-chip[data-subject]", "click", (e) => {
      const sub = e.target.closest("[data-subject]").dataset.subject;
      let cur = [...model.get("subjects")];
      cur = cur.includes(sub) ? cur.filter(x => x !== sub) : [...cur, sub];
      model.set("subjects", cur);
      model.save_changes();
    });

    // Regressor table — individual cell click
    const wireRegCell = (attr, trait) => {
      bindAll(`.mm-reg-cell[data-${attr}]`, "click", (e) => {
        const col = e.target.dataset[attr];
        if (!col) return;
        let cur = [...model.get(trait)];
        cur = cur.includes(col) ? cur.filter(x => x !== col) : [...cur, col];
        model.set(trait, cur);
        model.save_changes();
      });
    };

    // Regressor table — row label click (toggle all members in group)
    const wireRegRowLabel = (attr, trait) => {
      bindAll(`.mm-reg-row-label[data-${attr}-group]`, "click", (e) => {
        const lbl = e.target.closest(`[data-${attr}-group]`);
        if (!lbl) return;
        // camelCase: data-emission-members → dataset.emissionMembers
        const key     = attr.charAt(0).toUpperCase() + attr.slice(1);
        const members = JSON.parse(lbl.dataset[`${attr}Members`] || "[]");
        if (!members.length) return;
        let cur = new Set(model.get(trait));
        const allIn = members.every(m => cur.has(m));
        allIn ? members.forEach(m => cur.delete(m)) : members.forEach(m => cur.add(m));
        model.set(trait, [...cur]);
        model.save_changes();
      });
    };

    wireRegCell("emission", "emission_cols");
    wireRegRowLabel("emission", "emission_cols");
    wireRegCell("transition", "transition_cols");
    wireRegRowLabel("transition", "transition_cols");

    const commitFreezeInput = (input) => {
      if (!input) return;
      const state = input.dataset.state;
      const feature = input.dataset.feature;
      if (state == null || !feature) return;

      const current = JSON.parse(JSON.stringify(model.get("frozen_emissions") || {}));
      const rawValue = input.value.trim();

      if (!rawValue) {
        if (current[state]) {
          delete current[state][feature];
          if (Object.keys(current[state]).length === 0) {
            delete current[state];
          }
        }
      } else {
        const parsed = parseFloat(rawValue);
        if (!Number.isFinite(parsed)) {
          input.value = current[state] && Number.isFinite(current[state][feature]) ? String(current[state][feature]) : "";
          return;
        }
        if (!current[state]) current[state] = {};
        current[state][feature] = parsed;
      }

      model.set("frozen_emissions", current);
      model.save_changes();
    };

    bindAll(".mm-freeze-input", "change", (e) => {
      commitFreezeInput(e.target);
    });
    bindAll(".mm-freeze-input", "keydown", (e) => {
      if (e.key !== "Enter") return;
      e.preventDefault();
      commitFreezeInput(e.target);
      e.target.blur();
    });

    // Task selector
    bind("#inp-task", "change", (e) => {
      model.set("task", e.target.value);
      model.save_changes();
    });

    // Synchronized slider + number input pairs
    const syncSlider = (rangeId, numId, trait, parseFn) => {
      bind("#" + rangeId, "input", (e) => {
        const val = parseFn(e.target.value);
        const num = el.querySelector("#" + numId);
        if (num) num.value = val;
        model.set(trait, val);
        model.save_changes();
      });
      bind("#" + numId, "change", (e) => {
        const val = parseFn(e.target.value);
        const rng = el.querySelector("#" + rangeId);
        if (rng) rng.value = val;
        model.set(trait, val);
        model.save_changes();
      });
    };

    syncSlider("inp-k-range",         "inp-k-num",         "K",         parseInt);
    syncSlider("inp-tau-range",        "inp-tau-num",        "tau",       parseInt);
    syncSlider("inp-lapse-max-range",  "inp-lapse-max-num",  "lapse_max", parseFloat);

    // Lapse checkbox
    bind("#inp-lapse", "change", (e) => {
      model.set("lapse", e.target.checked);
      model.save_changes();
    });

    // Alias field + save button
    const commitAlias = ({ saveClick = false } = {}) => {
      model.set("alias", aliasDraft);
      if (saveClick) {
        model.set("save_alias_clicks", model.get("save_alias_clicks") + 1);
      }
      model.save_changes();
      aliasDirty = false;
    };

    bind("#inp-alias", "input", (e) => {
      aliasDraft = e.target.value;
      aliasDirty = true;
    });
    bind("#inp-alias", "change", (e) => {
      aliasDraft = e.target.value;
      commitAlias();
    });
    bind("#inp-alias", "keydown", (e) => {
      if (e.key !== "Enter") return;
      e.preventDefault();
      aliasDraft = e.target.value;
      commitAlias({ saveClick: true });
    });
    bind("#btn-save-alias", "click", () => {
      commitAlias({ saveClick: true });
    });

    // Run button
    bind("#btn-run", "click", () => {
      if (model.get("is_running")) {
        return;
      }
      model.set("run_fit_clicks", model.get("run_fit_clicks") + 1);
      model.save_changes();
    });
  };

  updateUI();
  model.on("change", updateUI);
}

export default { render };
