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

// ── Main render ───────────────────────────────────────────────────────────────

function render({ model, el }) {
  const containerId = "mm-" + Math.random().toString(36).substring(7);

  const updateUI = () => {
    const existingVal      = model.get("existing_model");
    const is2afc           = model.get("is_2afc");
    const modelType        = model.get("model_type");
    const currentTask      = model.get("task");
    const mode             = model.get("ui_mode");
    const KList            = model.get("k_options");
    const currentK         = model.get("K");
    const subjectsList     = model.get("subjects_list");
    const currentSubjects  = model.get("subjects");
    const currentEmission  = model.get("emission_cols");
    const currentTransition = model.get("transition_cols");
    const currentTau       = model.get("tau");
    const currentLapse     = model.get("lapse");
    const currentLapseMax  = model.get("lapse_max");
    const currentAlias     = model.get("alias");
    const existingInfo     = model.get("existing_models_info");
    const emissionGroups   = model.get("emission_groups");
    const transitionGroups = model.get("transition_groups");

    // ── Shell ───────────────────────────────────────────────────────────────
    let html = `
      <div class="mm-container" id="${containerId}">
        <div class="mm-header">
          <div class="mm-task-selector">
            <label class="mm-label inline">Task:</label>
            <select id="inp-task" class="mm-input small">
              <option value="MCDR" ${currentTask === "MCDR" ? "selected" : ""}>MCDR</option>
              <option value="2AFC" ${currentTask === "2AFC" ? "selected" : ""}>2AFC</option>
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
                  <th>Tau</th>
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
                      <td>${info.tau}</td>
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
            <input type="text" id="inp-alias" class="mm-input"
                   placeholder="e.g. my_best_fit" value="${currentAlias}">
            <button class="mm-btn-secondary" id="btn-save-alias">Save</button>
          </div>
          <button class="mm-btn-run" id="btn-run">RUN FIT</button>
        </div>
      </div>
    </div>
    `;

    el.innerHTML = html;

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
      const row = e.target.closest(".mm-tr");
      if (!row) return;
      el.querySelectorAll(".mm-tr").forEach(r => r.classList.remove("selected"));
      row.classList.add("selected");
      model.set("existing_model", row.dataset.model);
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
    bind("#inp-alias", "input", (e) => {
      model.set("alias", e.target.value);
      model.save_changes();
    });
    bind("#btn-save-alias", "click", (e) => {
      model.set("save_alias_clicks", model.get("save_alias_clicks") + 1);
      model.save_changes();
      const btn = e.target;
      const orig = btn.innerText;
      btn.innerText = "Saved!";
      btn.classList.add("saved");
      setTimeout(() => { btn.innerText = orig; btn.classList.remove("saved"); }, 1000);
    });

    // Run button
    bind("#btn-run", "click", (e) => {
      model.set("run_fit_clicks", model.get("run_fit_clicks") + 1);
      model.save_changes();
      const btn = e.target;
      const orig = btn.innerText;
      btn.innerText = "FITTING...";
      btn.classList.add("running");
      setTimeout(() => { btn.innerText = orig; btn.classList.remove("running"); }, 800);
    });
  };

  updateUI();
  model.on("change", updateUI);
}

export default { render };
