import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import traitlets
    import anywidget
    from pathlib import Path
    from typing import Any, Dict

    try:
        import tomllib  # py3.11+
    except ImportError:  # pragma: no cover
        import tomli as tomllib  # type: ignore

    try:
        import tomli_w
    except Exception:  # pragma: no cover
        tomli_w = None


    class TomlConfigEditor(anywidget.AnyWidget):
        """
        A lightweight TOML editor widget.

        Frontend sends commands via (command, command_payload, command_nonce).
        Python side executes load/save and updates (data, status, path).
        """

        # ---- Synced state
        data = traitlets.Dict(default_value={}).tag(sync=True)
        path = traitlets.Unicode(default_value="").tag(sync=True)
        name = traitlets.Unicode(default_value="config").tag(sync=True)
        status = traitlets.Unicode(default_value="").tag(sync=True)

        # ---- Command channel (frontend -> python)
        command = traitlets.Unicode(default_value="").tag(sync=True)
        command_payload = traitlets.Dict(default_value={}).tag(sync=True)
        command_nonce = traitlets.Int(default_value=0).tag(sync=True)

        _css = r"""
        :host { display: block; }

        .tce{
          font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
          color: #111;
        }

        .topbar{
          display:flex; gap:10px; align-items:center; flex-wrap:wrap;
          padding:12px; border:1px solid #e5e5e5; border-radius:14px;
          background: linear-gradient(180deg, #ffffff, #fbfbfb);
          box-shadow: 0 4px 18px rgba(0,0,0,0.05);
        }

        .pill{
          display:flex; align-items:center; gap:8px;
          padding:8px 10px; border:1px solid #e2e2e2; border-radius:999px; background:#fff;
        }

        .label{ font-size:12px; color:#555; }

        .input{
          border:none; background:transparent;
          padding:6px 8px; min-width:240px; outline:none; color: inherit;
        }
        .input:focus{ outline:none; box-shadow:none; }

        .btn{
          border:1px solid #d6d6d6; background:#fff; border-radius:12px;
          padding:9px 12px; cursor:pointer; transition: transform .05s, background .15s, border-color .15s;
          display:flex; align-items:center; gap:8px;
          user-select:none;
        }
        .btn:hover{ background:#f6f6f6; }
        .btn:active{ transform: translateY(1px); }
        .btn.primary{ border-color:#c9d6e6; background:#eef5ff; }
        .btn.primary:hover{ background:#e4efff; }
        .btn.danger:hover{ background:#fff1f1; border-color:#f0b3b3; }
        .btn:disabled{ opacity: 0.55; cursor: not-allowed; }

        .status{ margin-left:auto; font-size:12px; color:#666; }

        .tabs{
          margin-top:12px;
          display:flex; gap:8px; flex-wrap:wrap;
        }

        .tab{
          border:1px solid #e0e0e0; background:#fff; border-radius:999px;
          padding:8px 12px; cursor:pointer; font-weight:700; color:#333;
        }
        .tab.active{
          background:#111; color:#fff; border-color:#111;
        }

        .panel{
          margin-top:12px;
          border:1px solid #e7e7e7; border-radius:16px;
          padding:14px; background:#fff;
          box-shadow: 0 6px 24px rgba(0,0,0,0.04);
        }

        .sectionTitle{
          font-size:14px; font-weight:800; color:#111; margin:0 0 10px 0;
          display:flex; align-items:center; justify-content:space-between;
        }

        .card{
          border:1px solid #ededed; border-radius:14px; padding:10px 10px;
          background: linear-gradient(180deg, #fff, #fcfcfc);
          margin:8px 0;
        }

        .row{
          display:grid;
          grid-template-columns: 220px 1fr auto;
          gap:10px;
          align-items:center;
          padding:6px 6px;
          border-radius:12px;
        }
        .row:hover{ background:#fafafa; }

        .k{
          font-weight:700; color:#222;
          overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
        }

        .v{ display:flex; gap:10px; align-items:center; min-width:0; }

        .text, .num, select, textarea{
          width:100%;
          border:1px solid #d8d8d8; border-radius:10px; padding:8px 10px;
          outline:none; color: inherit; background: #fff;
        }
        .text:focus, .num:focus, select:focus, textarea:focus{
          border-color:#9ab; box-shadow: 0 0 0 3px rgba(100,130,170,0.15);
        }

        textarea{
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        }

        .fold{
          cursor:pointer; user-select:none; display:flex; align-items:center; gap:8px;
          font-weight:800; color:#111;
          padding:6px 6px; border-radius:12px;
        }
        .fold:hover{ background:#f5f5f5; }

        .indent{ margin-left:16px; }

        .color{
          width:44px; height:36px; border:1px solid #d8d8d8; border-radius:10px; padding:0; background:#fff;
        }

        .hint{ font-size:12px; color:#777; margin-top:6px; }

        .addbox{
          display:grid;
          grid-template-columns: 1.2fr 1fr 1fr auto;
          gap:10px;
          align-items:center;
          padding:10px;
          border:1px dashed #ddd;
          border-radius:14px;
          background:#fff;
          margin:10px 0 14px 0;
        }
        .addboxTitle{
          font-size:12px; font-weight:800; color:#333; margin:0 0 6px 2px;
        }

        /* --- Dark mode --- */
        @media (prefers-color-scheme: dark){
          .tce{ color:#fff; }
          .topbar{
            border-color:#2b2b2b;
            background: linear-gradient(180deg, #1a1a1a, #141414);
            box-shadow: 0 8px 26px rgba(0,0,0,0.35);
          }
          .pill{ border-color:#2b2b2b; background:#121212; }
          .label{ color:#bdbdbd; }
          .status{ color:#bdbdbd; }

          .btn{ border-color:#2b2b2b; background:#141414; color:#fff; }
          .btn:hover{ background:#1b1b1b; }
          .btn.primary{ border-color:#2a3b55; background:#122033; }
          .btn.primary:hover{ background:#152a45; }
          .btn.danger:hover{ background:#2a1414; border-color:#5a2a2a; }

          .tab{ border-color:#2b2b2b; background:#141414; color:#fff; }
          .tab.active{ background:#fff; color:#111; border-color:#fff; }

          .panel{ border-color:#2b2b2b; background:#121212; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }
          .sectionTitle{ color:#fff; }

          .card{ border-color:#2b2b2b; background: linear-gradient(180deg, #141414, #101010); }
          .row:hover{ background:#171717; }
          .k{ color:#fff; }

          .text, .num, select, textarea{
            border-color:#2b2b2b; background:#0f0f0f; color:#fff;
          }
          .fold{ color:#fff; }
          .fold:hover{ background:#1a1a1a; }

          .hint{ color:#bdbdbd; }
          .addbox{ border-color:#2b2b2b; background:#141414; }
          .addboxTitle{ color:#d5d5d5; }
          .color{ border-color:#2b2b2b; background:#0f0f0f; }
        }
        """

        _esm = r"""
        function deepClone(x){ return JSON.parse(JSON.stringify(x)); }
        function isHexColor(s){ return typeof s === "string" && /^#[0-9A-Fa-f]{6}$/.test(s); }

        function setByPath(obj, path, value){
          const parts = path.split(".").filter(Boolean);
          let cur = obj;
          for(let i=0;i<parts.length-1;i++){
            const p = parts[i];
            if(typeof cur[p] !== "object" || cur[p] === null || Array.isArray(cur[p])) cur[p] = {};
            cur = cur[p];
          }
          cur[parts[parts.length-1]] = value;
        }

        function deleteByPath(obj, path){
          const parts = path.split(".").filter(Boolean);
          let cur = obj;
          for(let i=0;i<parts.length-1;i++){
            const p = parts[i];
            if(!(p in cur)) return;
            cur = cur[p];
            if(typeof cur !== "object" || cur === null) return;
          }
          delete cur[parts[parts.length-1]];
        }

        function getByPath(obj, path){
          const parts = path.split(".").filter(Boolean);
          let cur = obj;
          for(const p of parts){
            if(!cur || typeof cur !== "object") return undefined;
            cur = cur[p];
          }
          return cur;
        }

        function keysSorted(obj){
          return Object.keys(obj || {}).sort((a,b)=>a.localeCompare(b));
        }

        export default {
          render({ model, el }) {
            el.innerHTML = "";

            const root = document.createElement("div");
            root.className = "tce";
            el.appendChild(root);

            // ---- History (undo/redo)
            let history = [];
            let hIndex = -1;

            function pushHistory(snapshot){
              history = history.slice(0, hIndex + 1);
              history.push(snapshot);
              hIndex = history.length - 1;
            }
            function canUndo(){ return hIndex > 0; }
            function canRedo(){ return hIndex < history.length - 1; }
            function applySnapshot(snapshot){
              model.set("data", deepClone(snapshot));
              model.save_changes();
            }
            function resetHistoryToCurrent(){
              history = [];
              hIndex = -1;
              pushHistory(deepClone(model.get("data") || {}));
            }
            function ensureHistoryInit(){
              if(history.length === 0){
                pushHistory(deepClone(model.get("data") || {}));
              }
            }

            // ---- UI state
            let activeTab = "root";
            const expanded = new Set(); // dict folds
            let expandedInitialized = false;

            function expandAllTablesByDefault(data){
              function walk(obj, basePath){
                for(const k of Object.keys(obj || {})){
                  const v = obj[k];
                  const p = basePath ? `${basePath}.${k}` : k;
                  if(v && typeof v === "object" && !Array.isArray(v)){
                    expanded.add(p);
                    walk(v, p);
                  }
                }
              }
              walk(data, "");
            }

            function sendCommand(type, payload){
              model.set("command", type);
              model.set("command_payload", payload || {});
              model.set("command_nonce", (model.get("command_nonce") || 0) + 1);
              model.save_changes();
            }

            function topLevelSplit(data){
              const rootScalars = {};
              const tables = {};
              for(const k of keysSorted(data)){
                const v = data[k];
                if(v && typeof v === "object" && !Array.isArray(v)){
                  tables[k] = v;
                } else {
                  rootScalars[k] = v;
                }
              }
              return { rootScalars, tables };
            }

            function commitChange(mutator){
              ensureHistoryInit();
              const data = deepClone(model.get("data") || {});
              mutator(data);
              model.set("data", data);
              model.save_changes();
              pushHistory(deepClone(data));
              renderAll();
            }

            function renderValueEditor(container, fullPath, key, value){
              if(typeof value === "boolean"){
                const chk = document.createElement("input");
                chk.type = "checkbox";
                chk.checked = !!value;
                chk.onchange = () => commitChange(d => setByPath(d, fullPath, chk.checked));
                container.appendChild(chk);
                return;
              }

              if(typeof value === "number"){
                const inp = document.createElement("input");
                inp.className = "num";
                inp.type = "number";
                inp.value = String(value);
                inp.onchange = () => {
                  const n = Number(inp.value);
                  commitChange(d => setByPath(d, fullPath, Number.isFinite(n) ? n : 0));
                };
                container.appendChild(inp);
                return;
              }

              if(typeof value === "string"){
                if(isHexColor(value) || key.toLowerCase().includes("color")){
                  const col = document.createElement("input");
                  col.type = "color";
                  col.className = "color";
                  col.value = isHexColor(value) ? value : "#000000";
                  col.oninput = () => commitChange(d => setByPath(d, fullPath, col.value));

                  const txt = document.createElement("input");
                  txt.type = "text";
                  txt.className = "text";
                  txt.value = value;
                  txt.onchange = () => commitChange(d => setByPath(d, fullPath, txt.value));

                  container.appendChild(col);
                  container.appendChild(txt);
                  return;
                }

                const inp = document.createElement("input");
                inp.type = "text";
                inp.className = "text";
                inp.value = value;
                inp.onchange = () => commitChange(d => setByPath(d, fullPath, inp.value));
                container.appendChild(inp);
                return;
              }

              if(Array.isArray(value)){
                const ta = document.createElement("textarea");
                ta.rows = 2;
                ta.value = JSON.stringify(value);
                ta.onchange = () => {
                  try{
                    const parsed = JSON.parse(ta.value);
                    if(!Array.isArray(parsed)) throw new Error("not array");
                    commitChange(d => setByPath(d, fullPath, parsed));
                    ta.style.borderColor = "";
                  } catch(e){
                    ta.style.borderColor = "#c00";
                  }
                };
                container.appendChild(ta);
                return;
              }

              const inp = document.createElement("input");
              inp.type = "text";
              inp.className = "text";
              inp.value = value == null ? "" : String(value);
              inp.onchange = () => commitChange(d => setByPath(d, fullPath, inp.value));
              container.appendChild(inp);
            }

            // --- Add UI (no prompts)
            function renderAddBox(basePath){
              const wrap = document.createElement("div");

              const title = document.createElement("div");
              title.className = "addboxTitle";
              title.textContent = `Add inside: ${basePath || "root"}`;
              wrap.appendChild(title);

              const box = document.createElement("div");
              box.className = "addbox";

              const key = document.createElement("input");
              key.className = "text";
              key.placeholder = "key_name";

              const type = document.createElement("select");
              type.innerHTML = `
                <option value="string">string</option>
                <option value="number">number</option>
                <option value="boolean">boolean</option>
                <option value="color">color</option>
                <option value="table">table</option>
              `;

              const val = document.createElement("input");
              val.className = "text";
              val.placeholder = "value";

              function syncValUI(){
                const t = type.value;
                if(t === "table"){
                  val.disabled = true;
                  val.value = "";
                  val.placeholder = "(empty)";
                } else if(t === "boolean"){
                  val.disabled = false;
                  val.value = "false";
                  val.placeholder = "true | false";
                } else if(t === "number"){
                  val.disabled = false;
                  val.value = "0";
                  val.placeholder = "0";
                } else if(t === "color"){
                  val.disabled = false;
                  val.value = "#000000";
                  val.placeholder = "#RRGGBB";
                } else {
                  val.disabled = false;
                  val.value = "";
                  val.placeholder = "text";
                }
              }
              type.onchange = syncValUI;
              syncValUI();

              const addBtn = document.createElement("button");
              addBtn.className = "btn primary";
              addBtn.textContent = "Add";

              addBtn.onclick = () => {
                const k = (key.value || "").trim();
                if(!k){ alert("Missing key name."); return; }
                const full = basePath ? `${basePath}.${k}` : k;

                commitChange(d => {
                  if(getByPath(d, full) !== undefined){ alert("That key already exists."); return; }
                  const t = type.value;
                  if(t === "table"){
                    setByPath(d, full, {});
                    expanded.add(full);
                  } else if(t === "boolean"){
                    const vv = String(val.value).toLowerCase().trim();
                    setByPath(d, full, vv === "true");
                  } else if(t === "number"){
                    const n = Number(val.value);
                    setByPath(d, full, Number.isFinite(n) ? n : 0);
                  } else if(t === "color"){
                    const c = String(val.value).trim();
                    setByPath(d, full, isHexColor(c) ? c : "#000000");
                  } else {
                    setByPath(d, full, String(val.value));
                  }
                });

                key.value = "";
                syncValUI();
              };

              box.appendChild(key);
              box.appendChild(type);
              box.appendChild(val);
              box.appendChild(addBtn);

              wrap.appendChild(box);
              return wrap;
            }

            function renderObjectCard(obj, basePath, titleText){
              const card = document.createElement("div");
              card.className = "card";

              const header = document.createElement("div");
              header.className = "sectionTitle";
              header.textContent = titleText;
              card.appendChild(header);

              card.appendChild(renderAddBox(basePath));

              const ks = keysSorted(obj);
              if(ks.length === 0){
                const empty = document.createElement("div");
                empty.className = "hint";
                empty.textContent = "No keys yet.";
                card.appendChild(empty);
                return card;
              }

              for(const k of ks){
                const v = obj[k];
                const fullPath = basePath ? `${basePath}.${k}` : k;

                const isObj = v && typeof v === "object" && !Array.isArray(v);
                if(isObj){
                  const open = expanded.has(fullPath);

                  const foldRow = document.createElement("div");
                  foldRow.style.display = "flex";
                  foldRow.style.justifyContent = "space-between";
                  foldRow.style.alignItems = "center";
                  foldRow.style.padding = "4px 6px";

                  const fold = document.createElement("div");
                  fold.className = "fold";
                  fold.textContent = `${open ? "▾" : "▸"} ${k}`;
                  fold.onclick = () => {
                    if(expanded.has(fullPath)) expanded.delete(fullPath);
                    else expanded.add(fullPath);
                    renderAll();
                  };

                  const del = document.createElement("button");
                  del.className = "btn danger";
                  del.style.padding = "8px 10px";
                  del.textContent = "🗑";
                  del.title = "Delete";
                  del.onclick = () => commitChange(d => deleteByPath(d, fullPath));

                  foldRow.appendChild(fold);
                  foldRow.appendChild(del);
                  card.appendChild(foldRow);

                  if(open){
                    const inner = renderObjectCard(v, fullPath, "Contents");
                    inner.classList.add("indent");
                    card.appendChild(inner);
                  }
                  continue;
                }

                const row = document.createElement("div");
                row.className = "row";

                const keyEl = document.createElement("div");
                keyEl.className = "k";
                keyEl.textContent = k;

                const valEl = document.createElement("div");
                valEl.className = "v";
                renderValueEditor(valEl, fullPath, k, v);

                const del = document.createElement("button");
                del.className = "btn danger";
                del.style.padding = "8px 10px";
                del.textContent = "🗑";
                del.title = "Delete";
                del.onclick = () => commitChange(d => deleteByPath(d, fullPath));

                row.appendChild(keyEl);
                row.appendChild(valEl);
                row.appendChild(del);
                card.appendChild(row);
              }

              return card;
            }

            // DOM nodes
            const topbar = document.createElement("div");
            topbar.className = "topbar";

            const namePill = document.createElement("div");
            namePill.className = "pill";
            const nameLabel = document.createElement("span");
            nameLabel.className = "label";
            nameLabel.textContent = "Name";
            const nameInput = document.createElement("input");
            nameInput.className = "input";
            nameInput.style.minWidth = "160px";
            nameInput.value = model.get("name") || "config";
            nameInput.onchange = () => {
              model.set("name", nameInput.value);
              model.save_changes();
            };
            namePill.appendChild(nameLabel);
            namePill.appendChild(nameInput);

            const pathPill = document.createElement("div");
            pathPill.className = "pill";
            const pathLabel = document.createElement("span");
            pathLabel.className = "label";
            pathLabel.textContent = "Path";
            const pathInput = document.createElement("input");
            pathInput.className = "input";
            pathInput.value = model.get("path") || "";
            pathInput.placeholder = "config.toml";
            pathPill.appendChild(pathLabel);
            pathPill.appendChild(pathInput);

            const openBtn = document.createElement("button");
            openBtn.className = "btn primary";
            openBtn.textContent = "Open";

            function openFromPath(){
              const p = (pathInput.value || "").trim();
              if(!p){
                alert("Please enter a path to a TOML file.");
                return;
              }
              model.set("path", p);
              model.save_changes();
              sendCommand("load", { path: p });
            }
            openBtn.onclick = openFromPath;
            pathInput.addEventListener("keydown", (e) => {
              if(e.key === "Enter") openFromPath();
            });

            const saveBtn = document.createElement("button");
            saveBtn.className = "btn primary";
            saveBtn.textContent = "Save";
            saveBtn.onclick = () => {
              const p = (model.get("path") || pathInput.value || "").trim();
              if(!p){
                alert("Please enter a path to save the TOML file.");
                return;
              }
              const snapshot = deepClone(model.get("data") || {});
              sendCommand("save", { path: p, data: snapshot });
            };

            const undoBtn = document.createElement("button");
            undoBtn.className = "btn";
            undoBtn.textContent = "Undo";
            undoBtn.onclick = () => {
              ensureHistoryInit();
              if(!canUndo()) return;
              hIndex -= 1;
              applySnapshot(history[hIndex]);
              renderAll();
            };

            const redoBtn = document.createElement("button");
            redoBtn.className = "btn";
            redoBtn.textContent = "Redo";
            redoBtn.onclick = () => {
              ensureHistoryInit();
              if(!canRedo()) return;
              hIndex += 1;
              applySnapshot(history[hIndex]);
              renderAll();
            };

            const status = document.createElement("div");
            status.className = "status";

            topbar.appendChild(namePill);
            topbar.appendChild(pathPill);
            topbar.appendChild(openBtn);
            topbar.appendChild(saveBtn);
            topbar.appendChild(undoBtn);
            topbar.appendChild(redoBtn);
            topbar.appendChild(status);

            const tabs = document.createElement("div");
            tabs.className = "tabs";

            const panel = document.createElement("div");
            panel.className = "panel";

            root.appendChild(topbar);
            root.appendChild(tabs);
            root.appendChild(panel);

            function renderAll(){
              nameInput.value = model.get("name") || "config";
              pathInput.value = model.get("path") || "";
              status.textContent = model.get("status") || "";

              openBtn.disabled = !(pathInput.value || "").trim();

              ensureHistoryInit();
              undoBtn.style.opacity = canUndo() ? "1" : "0.5";
              undoBtn.style.pointerEvents = canUndo() ? "auto" : "none";
              redoBtn.style.opacity = canRedo() ? "1" : "0.5";
              redoBtn.style.pointerEvents = canRedo() ? "auto" : "none";

              tabs.innerHTML = "";
              panel.innerHTML = "";

              const data = model.get("data") || {};
              if(!expandedInitialized){
                expandAllTablesByDefault(data);
                expandedInitialized = true;
              }

              const { rootScalars, tables } = topLevelSplit(data);
              const tabNames = ["root", ...keysSorted(tables)];
              if(!tabNames.includes(activeTab)) activeTab = "root";

              for(const t of tabNames){
                const b = document.createElement("button");
                b.className = "tab" + (t === activeTab ? " active" : "");
                b.textContent = (t === "root") ? "root" : t;
                b.onclick = () => { activeTab = t; renderAll(); };
                tabs.appendChild(b);
              }

              if(activeTab === "root"){
                const title = document.createElement("div");
                title.className = "sectionTitle";
                title.textContent = "Root (non-table values)";
                panel.appendChild(title);
                panel.appendChild(renderObjectCard(rootScalars, "", "Root"));
              } else {
                const title = document.createElement("div");
                title.className = "sectionTitle";
                title.textContent = `Table: ${activeTab}`;
                panel.appendChild(title);
                panel.appendChild(renderObjectCard(tables[activeTab] || {}, activeTab, activeTab));
              }
            }

            // Sync Python -> UI
            model.on("change:data", () => {
              expandedInitialized = false;
              resetHistoryToCurrent();
              renderAll();
            });
            model.on("change:status", renderAll);
            model.on("change:path", renderAll);
            model.on("change:name", renderAll);

            resetHistoryToCurrent();
            renderAll();
          }
        };
        """

        def __init__(self, path: str = "config.toml", name: str = "config"):
            super().__init__()
            self.name = name
            self.path = str(Path(path).expanduser()) if path else ""
            self.status = "Ready."
            self.data = {}

            if self.path:
                self.load(self.path)

        def load(self, path: str) -> None:
            p = Path(path).expanduser()

            if not p.exists():
                self.data = {}
                self.status = f"File not found: {p}"
                return

            try:
                with p.open("rb") as f:
                    obj = tomllib.load(f)
                self.path = str(p)
                self.data = obj if isinstance(obj, dict) else {}
                self.status = f"Loaded: {p}"
            except Exception as e:
                self.data = {}
                self.status = f"Error loading TOML: {e}"

        def save(self, path: str) -> None:
            if tomli_w is None:
                self.status = "Install tomli-w to enable saving (pip install tomli-w)."
                return

            p = Path(path).expanduser()

            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                text = tomli_w.dumps(self.data)  # type: ignore[union-attr]
                p.write_text(text, encoding="utf-8")
                self.path = str(p)
                self.status = f"Saved: {p.resolve()}"
            except Exception as e:
                self.status = f"Error saving: {e} (path: {p.resolve()})"

        @traitlets.observe("command_nonce")
        def _on_command(self, change: Dict[str, Any]) -> None:
            cmd = self.command
            payload = self.command_payload or {}

            if cmd == "load":
                self.load(payload.get("path", self.path))

            elif cmd == "save":
                d = payload.get("data")
                if isinstance(d, dict):
                    self.data = d
                self.save(payload.get("path", self.path))

            else:
                if cmd:
                    self.status = f"Unknown command: {cmd}"

            # prevent accidental re-exec
            self.command = ""
            self.command_payload = {}

    return (TomlConfigEditor,)


@app.cell
def _(TomlConfigEditor, mo):
    editor = mo.ui.anywidget(TomlConfigEditor("config.toml", name="Mi Config"))
    editor
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
