// marimo-toml-editor — widget.js
// Vanilla ESM, no build step required.

// ---- Utilities ----------------------------------------------------------------

function deepClone(x) { return JSON.parse(JSON.stringify(x)); }
function isHexColor(s) { return typeof s === "string" && /^#[0-9A-Fa-f]{6}$/.test(s); }

function setByPath(obj, path, value) {
    const parts = path.split(".").filter(Boolean);
    let cur = obj;
    for (let i = 0; i < parts.length - 1; i++) {
        const p = parts[i];
        if (typeof cur[p] !== "object" || cur[p] === null || Array.isArray(cur[p])) cur[p] = {};
        cur = cur[p];
    }
    cur[parts[parts.length - 1]] = value;
}

function deleteByPath(obj, path) {
    const parts = path.split(".").filter(Boolean);
    let cur = obj;
    for (let i = 0; i < parts.length - 1; i++) {
        const p = parts[i];
        if (!(p in cur)) return;
        cur = cur[p];
        if (typeof cur !== "object" || cur === null) return;
    }
    delete cur[parts[parts.length - 1]];
}

function getByPath(obj, path) {
    const parts = path.split(".").filter(Boolean);
    let cur = obj;
    for (const p of parts) {
        if (!cur || typeof cur !== "object") return undefined;
        cur = cur[p];
    }
    return cur;
}

function keysSorted(obj) {
    return Object.keys(obj || {}).sort((a, b) => a.localeCompare(b));
}

function isShallowScalarDict(obj) {
    if (!obj || typeof obj !== "object" || Array.isArray(obj)) return false;
    return Object.values(obj).every(v => v === null || typeof v !== "object");
}

function valueTypeName(v) {
    if (v === null || v === undefined) return "null";
    if (Array.isArray(v)) return "arr";
    if (typeof v === "boolean") return "bool";
    if (typeof v === "number") return Number.isInteger(v) ? "int" : "float";
    if (typeof v === "string") return "str";
    if (typeof v === "object") return "dict";
    return "str";
}

function typeBadge(v) {
    const name = valueTypeName(v);
    const b = document.createElement("span");
    b.className = `type-badge tb-${name}`;
    const labels = { str: "str", int: "int", float: "f", bool: "bool", arr: "[]", dict: "{}", null: "∅" };
    b.textContent = labels[name] || name;
    return b;
}

function iconBtn(emoji, title, extraClass) {
    const b = document.createElement("button");
    b.className = "btn-icon" + (extraClass ? " " + extraClass : "");
    b.textContent = emoji;
    b.title = title;
    b.type = "button";
    return b;
}

// ---- Module entry -------------------------------------------------------------

export default {
    render({ model, el }) {
        el.innerHTML = "";
        const root = document.createElement("div");
        root.className = "tce";
        el.appendChild(root);

        // ---- History -----------------------------------------------------------------
        let history = [];
        let hIndex = -1;
        // Guards to prevent change:data from resetting history inappropriately
        let _localChange = false;      // set during commitChange (user edit)
        let _applyingSnapshot = false; // set during applySnapshot (undo/redo)

        function pushHistory(snapshot) {
            history = history.slice(0, hIndex + 1);
            history.push(snapshot);
            hIndex = history.length - 1;
        }
        function canUndo() { return hIndex > 0; }
        function canRedo() { return hIndex < history.length - 1; }

        function applySnapshot(snapshot) {
            _applyingSnapshot = true;
            model.set("data", deepClone(snapshot));
            model.save_changes();
            _applyingSnapshot = false;
        }

        function resetHistoryToCurrent() {
            history = [];
            hIndex = -1;
            pushHistory(deepClone(model.get("data") || {}));
        }

        function ensureHistoryInit() {
            if (history.length === 0) pushHistory(deepClone(model.get("data") || {}));
        }

        // ---- UI state ----------------------------------------------------------------
        let activeTab = "root";
        let searchQuery = "";
        const expanded = new Set();
        let expandedInitialized = false;
        let isDirty = false;

        function expandAllTablesByDefault(data) {
            function walk(obj, basePath) {
                for (const k of Object.keys(obj || {})) {
                    const v = obj[k];
                    const p = basePath ? `${basePath}.${k}` : k;
                    if (v && typeof v === "object" && !Array.isArray(v)) {
                        expanded.add(p);
                        walk(v, p);
                    }
                }
            }
            walk(data, "");
        }

        function sendCommand(type, payload) {
            model.set("command", type);
            model.set("command_payload", payload || {});
            model.set("command_nonce", (model.get("command_nonce") || 0) + 1);
            model.save_changes();
        }

        function topLevelSplit(data) {
            const rootScalars = {};
            const tables = {};
            for (const k of keysSorted(data)) {
                const v = data[k];
                if (v && typeof v === "object" && !Array.isArray(v)) tables[k] = v;
                else rootScalars[k] = v;
            }
            return { rootScalars, tables };
        }

        function markDirty() {
            isDirty = true;
            if (dirtyDot) dirtyDot.classList.add("visible");
        }
        function markClean() {
            isDirty = false;
            if (dirtyDot) dirtyDot.classList.remove("visible");
        }

        // commitChange: user made an edit in the UI
        // _localChange flag prevents the resulting change:data event from resetting history
        function commitChange(mutator) {
            ensureHistoryInit();
            const data = deepClone(model.get("data") || {});
            mutator(data);
            _localChange = true;
            model.set("data", data);
            model.save_changes();
            _localChange = false;
            pushHistory(deepClone(data));
            markDirty();
            renderAll();
        }

        // ---- Value editors -----------------------------------------------------------

        /** Rich list editor — one item per row, type-aware, reorder + delete. */
        function renderListEditor(container, fullPath, arr) {
            const wrap = document.createElement("div");
            wrap.className = "list-editor";

            const itemsDiv = document.createElement("div");
            itemsDiv.className = "list-items";
            wrap.appendChild(itemsDiv);

            function inferType(a) {
                if (!a || a.length === 0) return "string";
                const first = a.find(x => x !== null);
                if (first === undefined) return "string";
                if (typeof first === "boolean") return "boolean";
                if (typeof first === "number") return "number";
                return "string";
            }

            const current = deepClone(arr);

            current.forEach((item, idx) => {
                const row = document.createElement("div");
                row.className = "list-item";

                let inp;
                if (typeof item === "boolean") {
                    inp = document.createElement("select");
                    inp.innerHTML = `<option value="true">true</option><option value="false">false</option>`;
                    inp.value = String(item);
                    inp.onchange = () => commitChange(d => {
                        const a = deepClone(getByPath(d, fullPath) || []);
                        a[idx] = inp.value === "true";
                        setByPath(d, fullPath, a);
                    });
                } else if (typeof item === "number") {
                    inp = document.createElement("input");
                    inp.type = "number"; inp.className = "num"; inp.value = String(item);
                    inp.onchange = () => {
                        const n = Number(inp.value);
                        commitChange(d => {
                            const a = deepClone(getByPath(d, fullPath) || []);
                            a[idx] = Number.isFinite(n) ? n : 0;
                            setByPath(d, fullPath, a);
                        });
                    };
                } else {
                    inp = document.createElement("input");
                    inp.type = "text"; inp.className = "text";
                    inp.value = item == null ? "" : String(item);
                    inp.onchange = () => commitChange(d => {
                        const a = deepClone(getByPath(d, fullPath) || []);
                        a[idx] = inp.value;
                        setByPath(d, fullPath, a);
                    });
                }

                const upBtn = iconBtn("↑", "Move up");
                upBtn.disabled = idx === 0;
                upBtn.onclick = () => commitChange(d => {
                    const a = deepClone(getByPath(d, fullPath) || []);
                    [a[idx - 1], a[idx]] = [a[idx], a[idx - 1]];
                    setByPath(d, fullPath, a);
                });

                const downBtn = iconBtn("↓", "Move down");
                downBtn.disabled = idx === current.length - 1;
                downBtn.onclick = () => commitChange(d => {
                    const a = deepClone(getByPath(d, fullPath) || []);
                    [a[idx], a[idx + 1]] = [a[idx + 1], a[idx]];
                    setByPath(d, fullPath, a);
                });

                const delBtn = iconBtn("✕", "Remove", "danger");
                delBtn.onclick = () => commitChange(d => {
                    const a = deepClone(getByPath(d, fullPath) || []);
                    a.splice(idx, 1);
                    setByPath(d, fullPath, a);
                });

                row.appendChild(inp);
                row.appendChild(upBtn);
                row.appendChild(downBtn);
                row.appendChild(delBtn);
                itemsDiv.appendChild(row);
            });

            // Add item row
            const addRow = document.createElement("div");
            addRow.className = "list-add-row";

            const guessedType = inferType(current);
            let addInp;
            if (guessedType === "boolean") {
                addInp = document.createElement("select");
                addInp.className = "text";
                addInp.innerHTML = `<option value="true">true</option><option value="false">false</option>`;
            } else if (guessedType === "number") {
                addInp = document.createElement("input");
                addInp.type = "number"; addInp.className = "num"; addInp.placeholder = "0"; addInp.value = "0";
            } else {
                addInp = document.createElement("input");
                addInp.type = "text"; addInp.className = "text"; addInp.placeholder = "new item…";
            }

            const addBtn = document.createElement("button");
            addBtn.className = "btn primary"; addBtn.textContent = "+ Add"; addBtn.type = "button";
            addBtn.onclick = () => {
                let newVal;
                if (guessedType === "boolean") newVal = addInp.value === "true";
                else if (guessedType === "number") newVal = Number(addInp.value);
                else newVal = addInp.value;
                commitChange(d => {
                    const a = deepClone(getByPath(d, fullPath) || []);
                    a.push(newVal);
                    setByPath(d, fullPath, a);
                });
                if (guessedType === "string") addInp.value = "";
            };
            addInp.addEventListener("keydown", e => { if (e.key === "Enter") addBtn.click(); });

            addRow.appendChild(addInp);
            addRow.appendChild(addBtn);
            wrap.appendChild(addRow);
            container.appendChild(wrap);
        }

        /** Inline dict editor for shallow scalar dicts. */
        const INLINE_LIMIT = 5;

        function renderInlineDict(container, fullPath, obj) {
            const wrap = document.createElement("div");
            wrap.className = "inline-dict";

            for (const k of keysSorted(obj)) {
                const v = obj[k];
                const childPath = fullPath ? `${fullPath}.${k}` : k;

                const row = document.createElement("div");
                row.className = "inline-dict-row";

                const kEl = document.createElement("div");
                kEl.className = "inline-dict-key";
                kEl.title = k; kEl.textContent = k;

                const vEl = document.createElement("div");
                renderScalarEditor(vEl, childPath, k, v);

                const del = iconBtn("✕", "Delete", "danger");
                del.onclick = () => commitChange(d => deleteByPath(d, childPath));

                row.appendChild(kEl); row.appendChild(vEl); row.appendChild(del);
                wrap.appendChild(row);
            }

            // Compact add row
            const addRow = document.createElement("div");
            addRow.className = "inline-dict-row";
            addRow.style.cssText = "border-top:1px dashed var(--border);margin-top:4px;padding-top:6px;";

            const newKey = document.createElement("input");
            newKey.type = "text"; newKey.className = "text";
            newKey.placeholder = "new key"; newKey.style.fontSize = "11px";

            const newVal = document.createElement("input");
            newVal.type = "text"; newVal.className = "text";
            newVal.placeholder = "value"; newVal.style.fontSize = "11px";

            const addBtn = document.createElement("button");
            addBtn.className = "btn primary"; addBtn.textContent = "＋";
            addBtn.type = "button"; addBtn.title = "Add key";
            addBtn.style.cssText = "font-size:11px;padding:4px 8px;";
            addBtn.onclick = () => {
                const k = newKey.value.trim();
                if (!k || k.includes(".")) return;
                const full = fullPath ? `${fullPath}.${k}` : k;
                commitChange(d => {
                    if (getByPath(d, full) !== undefined) return;
                    const raw = newVal.value;
                    if (raw === "true" || raw === "false") setByPath(d, full, raw === "true");
                    else if (raw !== "" && !isNaN(Number(raw))) setByPath(d, full, Number(raw));
                    else setByPath(d, full, raw);
                });
                newKey.value = ""; newVal.value = "";
            };
            newVal.addEventListener("keydown", e => { if (e.key === "Enter") addBtn.click(); });

            addRow.appendChild(newKey); addRow.appendChild(newVal); addRow.appendChild(addBtn);
            wrap.appendChild(addRow);
            container.appendChild(wrap);
        }

        function renderScalarEditor(container, fullPath, key, value) {
            if (typeof value === "boolean") {
                const sel = document.createElement("select");
                sel.innerHTML = `<option value="true">true</option><option value="false">false</option>`;
                sel.value = String(value);
                sel.onchange = () => commitChange(d => setByPath(d, fullPath, sel.value === "true"));
                container.appendChild(sel);
                return;
            }

            if (typeof value === "number") {
                const inp = document.createElement("input");
                inp.className = "num"; inp.type = "number"; inp.value = String(value);
                inp.onchange = () => {
                    const n = Number(inp.value);
                    commitChange(d => setByPath(d, fullPath, Number.isFinite(n) ? n : 0));
                };
                container.appendChild(inp);
                return;
            }

            if (typeof value === "string") {
                if (isHexColor(value) || key.toLowerCase().includes("color")) {
                    const wrap = document.createElement("div");
                    wrap.className = "color-wrap";
                    const col = document.createElement("input");
                    col.type = "color"; col.className = "color";
                    col.value = isHexColor(value) ? value : "#000000";
                    const txt = document.createElement("input");
                    txt.type = "text"; txt.className = "text"; txt.value = value;
                    col.oninput = () => { txt.value = col.value; commitChange(d => setByPath(d, fullPath, col.value)); };
                    txt.onchange = () => commitChange(d => setByPath(d, fullPath, txt.value));
                    wrap.appendChild(col); wrap.appendChild(txt);
                    container.appendChild(wrap);
                    return;
                }
                const inp = document.createElement("input");
                inp.type = "text"; inp.className = "text"; inp.value = value;
                inp.onchange = () => commitChange(d => setByPath(d, fullPath, inp.value));
                container.appendChild(inp);
                return;
            }

            // null / unknown
            const inp = document.createElement("input");
            inp.type = "text"; inp.className = "text";
            inp.value = value == null ? "" : String(value);
            inp.onchange = () => commitChange(d => setByPath(d, fullPath, inp.value));
            container.appendChild(inp);
        }

        // ---- Add box ----------------------------------------------------------------

        function renderAddBox(basePath) {
            const wrap = document.createElement("div");
            const title = document.createElement("div");
            title.className = "addboxTitle";
            title.textContent = `Add to: ${basePath || "root"}`;
            wrap.appendChild(title);

            const box = document.createElement("div");
            box.className = "addbox";

            const key = document.createElement("input");
            key.className = "text"; key.placeholder = "key_name";

            const type = document.createElement("select");
            type.innerHTML = `
        <option value="string">string</option>
        <option value="number">number</option>
        <option value="boolean">boolean</option>
        <option value="color">color</option>
        <option value="array">array []</option>
        <option value="table">table {}</option>
      `;

            const val = document.createElement("input");
            val.className = "text"; val.placeholder = "value";

            function syncValUI() {
                const t = type.value;
                if (t === "table" || t === "array") {
                    val.disabled = true; val.value = "";
                    val.placeholder = t === "table" ? "(empty {})" : "(empty [])";
                } else if (t === "boolean") {
                    val.disabled = false; val.value = "false"; val.placeholder = "true | false";
                } else if (t === "number") {
                    val.disabled = false; val.value = "0"; val.placeholder = "0";
                } else if (t === "color") {
                    val.disabled = false; val.value = "#000000"; val.placeholder = "#RRGGBB";
                } else {
                    val.disabled = false; val.value = ""; val.placeholder = "text";
                }
            }
            type.onchange = syncValUI; syncValUI();

            const addBtn = document.createElement("button");
            addBtn.className = "btn primary"; addBtn.textContent = "＋ Add"; addBtn.type = "button";
            addBtn.onclick = () => {
                const k = (key.value || "").trim();
                if (!k || k.includes(".")) return;
                const full = basePath ? `${basePath}.${k}` : k;
                commitChange(d => {
                    if (getByPath(d, full) !== undefined) return;
                    const t = type.value;
                    if (t === "table") { setByPath(d, full, {}); expanded.add(full); }
                    else if (t === "array") { setByPath(d, full, []); }
                    else if (t === "boolean") { setByPath(d, full, val.value.toLowerCase().trim() === "true"); }
                    else if (t === "number") { const n = Number(val.value); setByPath(d, full, Number.isFinite(n) ? n : 0); }
                    else if (t === "color") { setByPath(d, full, isHexColor(val.value.trim()) ? val.value.trim() : "#000000"); }
                    else { setByPath(d, full, String(val.value)); }
                });
                key.value = ""; syncValUI();
            };

            box.appendChild(key); box.appendChild(type); box.appendChild(val); box.appendChild(addBtn);
            wrap.appendChild(box);
            return wrap;
        }

        // ---- Object card ------------------------------------------------------------

        function renderObjectCard(obj, basePath, titleText) {
            const card = document.createElement("div");
            card.className = "card";

            const header = document.createElement("div");
            header.className = "sectionTitle";
            header.textContent = titleText;
            card.appendChild(header);
            card.appendChild(renderAddBox(basePath));

            const ks = keysSorted(obj);
            const visible = searchQuery
                ? ks.filter(k => k.toLowerCase().includes(searchQuery.toLowerCase()))
                : ks;

            if (ks.length === 0) {
                const e = document.createElement("div"); e.className = "hint";
                e.textContent = "No keys yet. Use the Add row above.";
                card.appendChild(e); return card;
            }
            if (visible.length === 0) {
                const e = document.createElement("div"); e.className = "hint";
                e.textContent = `No keys match "${searchQuery}".`;
                card.appendChild(e); return card;
            }

            for (const k of visible) {
                const v = obj[k];
                const fullPath = basePath ? `${basePath}.${k}` : k;
                const isObj = v && typeof v === "object" && !Array.isArray(v);

                // Shallow dict → inline
                if (isObj && Object.keys(v).length <= INLINE_LIMIT && isShallowScalarDict(v)) {
                    const row = document.createElement("div");
                    row.className = "row"; row.style.alignItems = "start";
                    const keyEl = document.createElement("div");
                    keyEl.className = "k";
                    keyEl.appendChild(typeBadge(v));
                    keyEl.appendChild(document.createTextNode(" " + k));
                    const valEl = document.createElement("div"); valEl.className = "v";
                    renderInlineDict(valEl, fullPath, v);
                    const del = iconBtn("✕", "Delete", "danger");
                    del.onclick = () => commitChange(d => deleteByPath(d, fullPath));
                    row.appendChild(keyEl); row.appendChild(valEl); row.appendChild(del);
                    card.appendChild(row); continue;
                }

                // Deep dict → fold
                if (isObj) {
                    const open = expanded.has(fullPath);
                    const foldRow = document.createElement("div"); foldRow.className = "fold-row";
                    const fold = document.createElement("div"); fold.className = "fold";
                    fold.innerHTML = `${open ? "▾" : "▸"} <span>${k}</span>`;
                    fold.onclick = () => {
                        if (expanded.has(fullPath)) expanded.delete(fullPath);
                        else expanded.add(fullPath);
                        renderAll();
                    };
                    const del = iconBtn("✕", "Delete section", "danger");
                    del.onclick = () => commitChange(d => deleteByPath(d, fullPath));
                    foldRow.appendChild(fold); foldRow.appendChild(del); card.appendChild(foldRow);
                    if (open) {
                        const inner = renderObjectCard(v, fullPath, "Contents");
                        inner.classList.add("indent"); card.appendChild(inner);
                    }
                    continue;
                }

                // Array → list editor
                if (Array.isArray(v)) {
                    const row = document.createElement("div");
                    row.className = "row"; row.style.alignItems = "start";
                    const keyEl = document.createElement("div"); keyEl.className = "k";
                    keyEl.appendChild(typeBadge(v));
                    keyEl.appendChild(document.createTextNode(" " + k));
                    const valEl = document.createElement("div"); valEl.className = "v";
                    renderListEditor(valEl, fullPath, v);
                    const del = iconBtn("✕", "Delete", "danger");
                    del.onclick = () => commitChange(d => deleteByPath(d, fullPath));
                    row.appendChild(keyEl); row.appendChild(valEl); row.appendChild(del);
                    card.appendChild(row); continue;
                }

                // Scalar
                const row = document.createElement("div"); row.className = "row";
                const keyEl = document.createElement("div"); keyEl.className = "k";
                keyEl.appendChild(typeBadge(v));
                keyEl.appendChild(document.createTextNode(" " + k));
                const valEl = document.createElement("div"); valEl.className = "v";
                renderScalarEditor(valEl, fullPath, k, v);
                const del = iconBtn("✕", "Delete", "danger");
                del.onclick = () => commitChange(d => deleteByPath(d, fullPath));
                row.appendChild(keyEl); row.appendChild(valEl); row.appendChild(del);
                card.appendChild(row);
            }
            return card;
        }

        // ---- Raw tab ----------------------------------------------------------------

        // ---- JS TOML serializer (covers all types the editor supports) ---------------

        function tomlVal(v) {
            if (typeof v === "boolean") return String(v);
            if (typeof v === "number") return String(v);
            if (typeof v === "string") return JSON.stringify(v);
            if (Array.isArray(v)) return "[" + v.map(tomlVal).join(", ") + "]";
            return JSON.stringify(v); // fallback
        }

        function tomlSerialize(data) {
            // Two-pass: scalars first, then [tables]
            let out = "";
            const tables = {};
            for (const k of keysSorted(data)) {
                const v = data[k];
                if (v && typeof v === "object" && !Array.isArray(v)) {
                    tables[k] = v;
                } else {
                    out += `${k} = ${tomlVal(v)}\n`;
                }
            }
            for (const [sec, obj] of Object.entries(tables)) {
                out += `\n[${sec}]\n`;
                for (const k of keysSorted(obj)) {
                    const v = obj[k];
                    if (v && typeof v === "object" && !Array.isArray(v)) {
                        // Nested table: use dotted [sec.sub] header
                        out += `\n[${sec}.${k}]\n`;
                        for (const sk of keysSorted(v)) {
                            out += `${sk} = ${tomlVal(v[sk])}\n`;
                        }
                    } else {
                        out += `${k} = ${tomlVal(v)}\n`;
                    }
                }
            }
            return out.trim() + "\n";
        }

        function getTomlText() {
            // Prefer Python-generated text (via tomli-w) if available and fresh,
            // otherwise fall back to the JS serializer.
            return (model.get("toml_text") || "").trim() || tomlSerialize(model.get("data") || {});
        }

        function renderRawPanel() {
            const wrap = document.createElement("div");
            const note = document.createElement("div");
            note.className = "hint"; note.style.marginBottom = "8px";
            note.textContent = "TOML preview. Use Save to write to disk.";
            wrap.appendChild(note);

            const ta = document.createElement("textarea");
            ta.className = "raw-area";
            ta.value = getTomlText();
            ta.readOnly = true;
            wrap.appendChild(ta);

            const copyBtn = document.createElement("button");
            copyBtn.className = "btn"; copyBtn.style.marginTop = "8px";
            copyBtn.textContent = "📋 Copy"; copyBtn.type = "button";
            copyBtn.onclick = () => {
                navigator.clipboard.writeText(ta.value).then(() => {
                    copyBtn.textContent = "✅ Copied!";
                    setTimeout(() => { copyBtn.textContent = "📋 Copy"; }, 1500);
                });
            };
            wrap.appendChild(copyBtn);
            return wrap;
        }

        // ---- File System operations (Delegated to Python) ---------------------------

        const isMac = typeof navigator !== "undefined" && navigator.userAgent.includes("Mac");

        async function openFilePicker() {
            if (isMac) {
                // Let Python handle the native macOS file dialog perfectly
                sendCommand("mac_native_open");
            } else {
                // Standard fallback for other OSes
                const inp = document.createElement("input");
                inp.type = "file"; inp.accept = ".toml";
                inp.onchange = async () => {
                    if (!inp.files || !inp.files[0]) return;
                    const file = inp.files[0];
                    const content = await file.text();
                    sendCommand("load_raw", { content, name: file.name });
                };
                inp.click();
            }
        }

        async function saveFilePicker(saveAs = false) {
            const tomlText = getTomlText();

            if (saveAs && isMac) {
                // Native macOS Save As dialog via Python
                sendCommand("mac_native_save_as", { content: tomlText });
                return;
            }

            if (saveAs && !isMac) {
                // Fallback for Save As on non-Mac: trigger download
                const suggestedName = (model.get("name") || "config") + ".toml";
                const blob = new Blob([tomlText], { type: "text/plain" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url; a.download = suggestedName; a.click();
                URL.revokeObjectURL(url);
                model.set("status", "Downloaded as file.");
                model.save_changes();
                markClean();
                return;
            }

            // Normal Save (overwrite current file directly in Python)
            sendCommand("save_local", { content: tomlText });
            markClean();
        }



        // ---- DOM construction -------------------------------------------------------

        // Editable H1 title
        const titleEl = document.createElement("div");
        titleEl.className = "widget-title";
        titleEl.contentEditable = "true";
        titleEl.spellcheck = false;
        titleEl.textContent = model.get("name") || "config";
        titleEl.title = "Click to rename";
        titleEl.addEventListener("blur", () => {
            const newName = titleEl.textContent.trim() || "config";
            titleEl.textContent = newName;
            model.set("name", newName);
            model.save_changes();
        });
        titleEl.addEventListener("keydown", e => {
            if (e.key === "Enter") { e.preventDefault(); titleEl.blur(); }
        });

        // Top bar (just buttons + status)
        const topbar = document.createElement("div");
        topbar.className = "topbar";

        const openBtn = document.createElement("button");
        openBtn.className = "btn primary"; openBtn.type = "button";
        openBtn.textContent = "📂 Open";
        openBtn.onclick = () => openFilePicker();

        const saveBtn = document.createElement("button");
        saveBtn.className = "btn primary"; saveBtn.type = "button";
        const dirtyDot = document.createElement("span");
        dirtyDot.className = "dirty-dot";
        saveBtn.appendChild(dirtyDot);
        saveBtn.appendChild(document.createTextNode("💾 Save"));
        saveBtn.onclick = () => saveFilePicker(false);

        const saveAsBtn = document.createElement("button");
        saveAsBtn.className = "btn"; saveAsBtn.type = "button";
        saveAsBtn.textContent = "📄 Save As";
        saveAsBtn.onclick = () => saveFilePicker(true);

        const undoBtn = document.createElement("button");
        undoBtn.className = "btn"; undoBtn.type = "button";
        undoBtn.textContent = "↩ Undo";
        undoBtn.onclick = () => {
            ensureHistoryInit();
            if (!canUndo()) return;
            hIndex -= 1;
            applySnapshot(history[hIndex]);
            renderAll();
        };

        const redoBtn = document.createElement("button");
        redoBtn.className = "btn"; redoBtn.type = "button";
        redoBtn.textContent = "↪ Redo";
        redoBtn.onclick = () => {
            ensureHistoryInit();
            if (!canRedo()) return;
            hIndex += 1;
            applySnapshot(history[hIndex]);
            renderAll();
        };

        const status = document.createElement("div");
        status.className = "status";

        topbar.appendChild(openBtn);
        topbar.appendChild(saveBtn);
        topbar.appendChild(saveAsBtn);
        topbar.appendChild(undoBtn);
        topbar.appendChild(redoBtn);
        topbar.appendChild(status);

        // Tabs + persistent search box
        const tabs = document.createElement("div");
        tabs.className = "tabs";

        // searchBox is created once and re-appended on each render (never recreated)
        const searchBox = document.createElement("input");
        searchBox.type = "search";
        searchBox.className = "search-box";
        searchBox.placeholder = "🔍 Filter keys…";
        searchBox.oninput = () => { searchQuery = searchBox.value; renderAll(); };

        const panel = document.createElement("div");
        panel.className = "panel";

        root.appendChild(titleEl);
        root.appendChild(topbar);
        root.appendChild(tabs);
        root.appendChild(panel);

        // Keyboard shortcuts
        el.addEventListener("keydown", e => {
            const mod = e.metaKey || e.ctrlKey;
            if (mod && e.key === "s") { e.preventDefault(); saveBtn.click(); }
            if (mod && e.key === "z" && !e.shiftKey) { e.preventDefault(); undoBtn.click(); }
            if (mod && (e.key === "y" || (e.key === "z" && e.shiftKey))) { e.preventDefault(); redoBtn.click(); }
        });

        // ---- Main render ------------------------------------------------------------

        function renderAll() {
            // Sync title (only if not currently focused to avoid caret jump)
            if (document.activeElement !== titleEl) {
                titleEl.textContent = model.get("name") || "config";
            }

            // Status color
            const s = model.get("status") || "";
            status.textContent = s;
            status.className = "status";
            if (s.startsWith("Loaded") || s.startsWith("Saved") || s === "Ready.") status.classList.add("ok");
            else if (s.startsWith("Error") || s.startsWith("File not found") || s.startsWith("Install")) status.classList.add("err");

            undoBtn.disabled = !canUndo();
            redoBtn.disabled = !canRedo();

            const data = model.get("data") || {};
            if (!expandedInitialized) {
                expandAllTablesByDefault(data);
                expandedInitialized = true;
            }

            const { rootScalars, tables } = topLevelSplit(data);
            const tabNames = ["root", ...keysSorted(tables), "raw"];
            if (!tabNames.includes(activeTab)) activeTab = "root";

            // Rebuild tabs — capture searchBox focus state before clearing
            const searchHadFocus = document.activeElement === searchBox;
            const searchCursorPos = searchHadFocus ? searchBox.selectionStart : null;

            tabs.innerHTML = "";
            for (const t of tabNames) {
                const b = document.createElement("button");
                b.className = "tab" + (t === activeTab ? " active" : "");
                b.textContent = t === "raw" ? "{ } Raw" : t;
                b.type = "button";
                b.onclick = () => { activeTab = t; renderAll(); };
                tabs.appendChild(b);
            }
            tabs.appendChild(searchBox); // re-append the persistent search box

            // Restore search focus and cursor position
            if (searchHadFocus) {
                searchBox.focus();
                if (searchCursorPos !== null) {
                    searchBox.setSelectionRange(searchCursorPos, searchCursorPos);
                }
            }

            // Panel
            panel.innerHTML = "";
            if (activeTab === "raw") {
                panel.appendChild(renderRawPanel());
            } else if (activeTab === "root") {
                const title = document.createElement("div");
                title.className = "sectionTitle"; title.textContent = "Root (scalar values)";
                panel.appendChild(title);
                panel.appendChild(renderObjectCard(rootScalars, "", "Root"));
            } else {
                const title = document.createElement("div");
                title.className = "sectionTitle"; title.textContent = `[${activeTab}]`;
                panel.appendChild(title);
                panel.appendChild(renderObjectCard(tables[activeTab] || {}, activeTab, activeTab));
            }
        }

        // ---- Model observers --------------------------------------------------------

        model.on("change:data", () => {
            // Guard: user edit or undo/redo → just re-render, keep history intact
            if (_localChange || _applyingSnapshot) { renderAll(); return; }
            // External change from Python (e.g. programmatic load) → reset history
            expandedInitialized = false;
            resetHistoryToCurrent();
            markClean();
            renderAll();
        });
        model.on("change:status", renderAll);
        model.on("change:name", () => {
            if (document.activeElement !== titleEl) {
                titleEl.textContent = model.get("name") || "config";
            }
        });

        resetHistoryToCurrent();
        renderAll();
    }
};
