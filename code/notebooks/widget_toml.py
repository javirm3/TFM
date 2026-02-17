import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    import traitlets
    import anywidget
    from pathlib import Path

    try:
        import tomllib  # py3.11+
    except ImportError:
        import tomli as tomllib

    try:
        import tomli_w
    except Exception:
        tomli_w = None


    class TomlConfigEditor(anywidget.AnyWidget):
        data = traitlets.Dict().tag(sync=True)
        path = traitlets.Unicode("").tag(sync=True)
        name = traitlets.Unicode("config").tag(sync=True)
        status = traitlets.Unicode("").tag(sync=True)
        command = traitlets.Unicode("").tag(sync=True)
        command_payload = traitlets.Dict().tag(sync=True)
        command_nonce = traitlets.Int(0).tag(sync=True)

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

        function css(){
          return `
          .tce{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
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
          /* 👇 sin “contornos” exagerados */
          .input{
            border:none; background:transparent;
            padding:6px 8px; min-width:240px; outline:none;
          }
          .input:focus{ outline:none; box-shadow:none; }
          .btn{
            border:1px solid #d6d6d6; background:#fff; border-radius:12px;
            padding:9px 12px; cursor:pointer; transition: transform .05s, background .15s, border-color .15s;
            display:flex; align-items:center; gap:8px;
          }
          .btn:hover{ background:#f6f6f6; }
          .btn:active{ transform: translateY(1px); }
          .btn.primary{ border-color:#c9d6e6; background:#eef5ff; }
          .btn.primary:hover{ background:#e4efff; }
          .btn.danger:hover{ background:#fff1f1; border-color:#f0b3b3; }
          .status{ margin-left:auto; font-size:12px; color:#666; }
          .tabs{
            margin-top:12px;
            display:flex; gap:8px; flex-wrap:wrap;
          }
          /* 👇 tabs sin iconos */
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
          .v{
            display:flex; gap:10px; align-items:center; min-width:0;
          }
          .text, .num, select, textarea{
            width:100%;
            border:1px solid #d8d8d8; border-radius:10px; padding:8px 10px;
            outline:none;
          }
          .text:focus, .num:focus, select:focus, textarea:focus{
            border-color:#9ab; box-shadow: 0 0 0 3px rgba(100,130,170,0.15);
          }
          textarea{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
          .mini{
            border:1px solid #d6d6d6; background:#fff; border-radius:10px;
            padding:7px 10px; cursor:pointer;
          }
          .mini:hover{ background:#f6f6f6; }
          .fold{
            cursor:pointer; user-select:none; display:flex; align-items:center; gap:8px;
            font-weight:800; color:#111;
            padding:6px 6px; border-radius:12px;
          }
          .fold:hover{ background:#f5f5f5; }
          .indent{ margin-left:16px; }
          .color{ width:44px; height:36px; border:1px solid #d8d8d8; border-radius:10px; padding:0; background:#fff; }
          .hint{ font-size:12px; color:#777; margin-top:6px; }

          /* Add-box (in-widget inputs) */
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
          .addbox .btn{ justify-content:center; }
          .addboxTitle{
            font-size:12px; font-weight:800; color:#333; margin:0 0 6px 2px;
          }
          `;
        }

        export default {
          render({ model, el }) {
            el.innerHTML = "";
            const style = document.createElement("style");
            style.textContent = css();
            el.appendChild(style);

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
              // expand top-level tables and their nested tables by default once
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
              title.textContent = `Añadir dentro de: ${basePath || "root"}`;
              wrap.appendChild(title);

              const box = document.createElement("div");
              box.className = "addbox";

              const key = document.createElement("input");
              key.className = "text";
              key.placeholder = "nombre_clave";

              const type = document.createElement("select");
              type.innerHTML = `
                <option value="string">string</option>
                <option value="number">number</option>
                <option value="boolean">boolean</option>
                <option value="color">color</option>
                <option value="table">subtabla</option>
              `;

              const val = document.createElement("input");
              val.className = "text";
              val.placeholder = "valor";
              val.value = "";

              function syncValUI(){
                const t = type.value;
                if(t === "table"){
                  val.disabled = true;
                  val.value = "";
                  val.placeholder = "(vacío)";
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
                  val.placeholder = "texto";
                }
              }
              type.onchange = syncValUI;
              syncValUI();

              const addBtn = document.createElement("button");
              addBtn.className = "btn primary";
              addBtn.textContent = "Añadir";

              addBtn.onclick = () => {
                const k = (key.value || "").trim();
                if(!k){ alert("Falta el nombre del campo."); return; }
                const full = basePath ? `${basePath}.${k}` : k;

                commitChange(d => {
                  if(getByPath(d, full) !== undefined){ alert("Esa clave ya existe."); return; }
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

                // reset
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

              // Add-box inside widget
              card.appendChild(renderAddBox(basePath));

              const ks = keysSorted(obj);
              if(ks.length === 0){
                const empty = document.createElement("div");
                empty.className = "hint";
                empty.textContent = "No hay claves aquí todavía.";
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
                  del.title = "Eliminar";
                  del.onclick = () => commitChange(d => deleteByPath(d, fullPath));

                  foldRow.appendChild(fold);
                  foldRow.appendChild(del);
                  card.appendChild(foldRow);

                  if(open){
                    const inner = renderObjectCard(v, fullPath, "Contenido");
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
                del.title = "Eliminar";
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
            nameLabel.textContent = "Nombre";
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
            pathLabel.textContent = "Ruta";
            const pathInput = document.createElement("input");
            pathInput.className = "input";
            pathInput.value = model.get("path") || "";
            pathInput.placeholder = "config.toml";
            pathPill.appendChild(pathLabel);
            pathPill.appendChild(pathInput);
            const openBtn = document.createElement("button");
            openBtn.className = "btn primary";
            openBtn.textContent = "Abrir";
            openBtn.onclick = () => {
              const p = pathInput.value;
              model.set("path", p);
              model.save_changes();
              sendCommand("load", { path: p });
            };

            const saveBtn = document.createElement("button");
            saveBtn.className = "btn primary";
            saveBtn.textContent = "Guardar";
            saveBtn.onclick = () => {
              const p = model.get("path") || pathInput.value;
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
                title.textContent = "Root (valores no-tabla)";
                panel.appendChild(title);
                panel.appendChild(renderObjectCard(rootScalars, "", "Root"));
              } else {
                const title = document.createElement("div");
                title.className = "sectionTitle";
                title.textContent = `Tabla: ${activeTab}`;
                panel.appendChild(title);
                panel.appendChild(renderObjectCard(tables[activeTab] || {}, activeTab, activeTab));
              }
            }

            // Sync Python -> UI
            model.on("change:data", () => {
              expandedInitialized = false; // re-expand for new file
              resetHistoryToCurrent();
              renderAll();
            });
            model.on("change:status", renderAll);
            model.on("change:path", renderAll);
            model.on("change:name", renderAll);

            // init history now
            resetHistoryToCurrent();
            renderAll();
          }
        };
        """

        def __init__(self, path="config.toml", name="config"):
            super().__init__()
            self.path = path
            self.name = name
            self.data = {}
            self.status = "Listo."
            self.on_msg(self._handle_msg)
            if path:
                self.load(path)

        def load(self, path: str):
            p = Path(path).expanduser()
            if not p.exists():
                self.status = f"No existe: {p}"
                self.data = {}
                return
            try:
                with open(p, "rb") as f:
                    obj = tomllib.load(f)
                self.path = str(p)
                self.data = obj
                self.status = f"Cargado: {p}"
            except Exception as e:
                self.status = f"Error cargando TOML: {e}"
                self.data = {}

        def save(self, path: str):
            if tomli_w is None:
                self.status = "Instala tomli-w para poder guardar (pip install tomli-w)."
                return

            p = Path(path).expanduser()
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                text = tomli_w.dumps(self.data)
                p.write_text(text, encoding="utf-8")
                self.path = str(p)
                self.status = f"Guardado: {p.resolve()}"  # 🔥 ruta real
            except Exception as e:
                self.status = f"Error guardando: {e} (ruta: {p.resolve()})"
            mo.md(f"SAVED TO: {p.resolve()}")

        def _handle_msg(self, msg, *args, **kwargs):
            # marimo/ipywidgets pueden envolver el payload en content/data
            payload = None

            if isinstance(msg, dict):
                # 1) ipywidgets style: msg["content"]["data"]
                content = msg.get("content")
                if isinstance(content, dict):
                    data = content.get("data")
                    if isinstance(data, dict):
                        payload = data
                    elif isinstance(content, dict):
                        # 2) a veces cae directo en content
                        payload = content

                # 3) a veces cae directo en msg
                if payload is None and "type" in msg:
                    payload = msg

            if not isinstance(payload, dict):
                return

            t = payload.get("type")
            if t is None:
                return

            if t == "load":
                self.load(payload.get("path", self.path))

            elif t == "save":
                if isinstance(payload.get("data"), dict):
                    self.data = payload["data"]
                self.save(payload.get("path", self.path))

            else:
                self.status = f"Mensaje desconocido: {t}"
        @traitlets.observe("command_nonce")
        def _on_command(self, change):
            cmd = self.command
            payload = self.command_payload or {}

            if cmd == "load":
                self.load(payload.get("path", self.path))

            elif cmd == "save":
                # snapshot del frontend (evita race)
                d = payload.get("data")
                if isinstance(d, dict):
                    self.data = d
                self.save(payload.get("path", self.path))

            # limpia comando para no re-ejecutar accidentalmente
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
