import os, json, time, requests
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from minio import Minio
from minio.error import S3Error

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "256"))

UPLOAD_DIR = Path("/tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")  # local|minio

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "cui-uploads")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".log", ".xml", ".html", ".htm", ".py", ".js", ".ts", ".css",
    ".java", ".kt", ".sql", ".ini", ".cfg", ".conf"
}

MAX_FILE_CHARS = int(os.getenv("MAX_FILE_CHARS", "40000"))

def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )

def put_object_minio(
    obj_name: str,
    data: bytes,
    content_type: str | None = None
) -> dict:
    client = get_minio_client()

    try:
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
    except S3Error:
        pass

    import io
    client.put_object(
        MINIO_BUCKET,
        obj_name,
        io.BytesIO(data),
        length=len(data),
        content_type=content_type or "application/octet-stream",
    )

    public_url = f"http://localhost:9000/{MINIO_BUCKET}/{obj_name}"

    return {"bucket": MINIO_BUCKET, "object": obj_name, "url": public_url}

def parse_export_intent(user_text: str) -> str | None:
    """
    Detect requests like:
      - "export last answer as csv"
      - "download as txt"
      - "save it as a csv file"
    Returns: "csv" | "txt" | None
    """
    t = (user_text or "").lower()

    export_words = ["export", "download", "save as", "save it as", "as a file", "to a file"]
    if not any(w in t for w in export_words):
        return None

    if "csv" in t:
        return "csv"
    if "txt" in t or "text file" in t or "plain text" in t:
        return "txt"
    return None


def last_assistant_text(history: list) -> str | None:
    """
    Returns the last assistant message content from history.
    """
    if not history:
        return None
    for m in reversed(history):
        if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"].strip():
            return m["content"]
    return None


def strip_code_fences(s: str) -> str:
    """
    Removes simple ```...``` fences if the model adds them.
    """
    if not s:
        return s
    s = s.strip()
    if s.startswith("```"):
        # remove first fence line
        parts = s.splitlines()
        # drop first line and possibly last line if it's ```
        if len(parts) >= 2 and parts[-1].strip().startswith("```"):
            parts = parts[1:-1]
        else:
            parts = parts[1:]
        s = "\n".join(parts).strip()
    return s


def convert_text_to_csv_via_model(text: str) -> str:
    """
    Uses the model to convert arbitrary text to CSV.
    If not naturally tabular, it must return single-column CSV:
      text
      "<full text>"
    """
    system = {
        "role": "system",
        "content": (
            "Convert the provided content into CSV.\n"
            "Rules:\n"
            "- Output ONLY raw CSV. No markdown, no code fences, no commentary.\n"
            "- Use commas as separators.\n"
            "- Always include a header row.\n"
            "- If the content is not naturally a table, output a single-column CSV:\n"
            "  header: text\n"
            "  one row containing the full content (properly quoted as CSV if needed).\n"
        ),
    }
    user = {"role": "user", "content": text}

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [system, user],
        "stream": False,
        "options": {"num_predict": max(NUM_PREDICT, 1024)},
    }

    r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=(10, 600))
    r.raise_for_status()
    out = r.json()
    csv_text = (out.get("message") or {}).get("content") or ""
    return strip_code_fences(csv_text).strip()


def save_export_to_minio(ext: str, data: bytes) -> dict:
    """
    Saves export bytes to MinIO and returns file info for UI.
    """
    ts = int(time.time())
    obj_name = f"export_{ts}_{int(time.time()*1000)}.{ext}"
    content_type = "text/plain" if ext == "txt" else "text/csv"
    stored = put_object_minio(obj_name, data, content_type=content_type)
    return {
        "format": ext,
        "name": obj_name,
        "url": stored["url"],
        "bucket": stored["bucket"],
        "object": stored["object"],
    }

def decode_bytes(data: bytes, max_chars: int) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "iso-8859-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            text = None
    if text is None:
        text = data.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + "\n\n[...truncated...]"
    return text

def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS

def read_text_file(path: Path, max_chars: int) -> str:
    """
    Best-effort decode with common encodings and truncate to max_chars.
    """
    data = path.read_bytes()

    for enc in ("utf-8", "utf-8-sig", "cp1252", "iso-8859-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            text = None
    if text is None:
        # Last resort: decode with replacement
        text = data.decode("utf-8", errors="replace")

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[...truncated...]"

    return text

def ollama_chat_stream(payload: dict):
    """
    Calls Ollama /api/chat with stream=true and yields SSE events containing text chunks.
    """
    with requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        stream=True,
        timeout=(10, 600),
    ) as r:
        r.raise_for_status()

        # Ollama streams JSON lines. Each line is a JSON object.
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # Typical chunk: {"message":{"role":"assistant","content":"..."}, "done": false, ...}
            chunk = ""
            msg = obj.get("message") or {}
            if isinstance(msg, dict):
                chunk = msg.get("content") or ""

            done = bool(obj.get("done", False))

            if chunk:
                # SSE data event
                yield f"data: {json.dumps({'type':'delta','text':chunk})}\n\n"

            if done:
                yield f"data: {json.dumps({'type':'done'})}\n\n"
                break

app = Flask(__name__, static_folder="static", static_url_path="/static")

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.get("/health")
def health():
    return {"ok": True, "ollama": OLLAMA_BASE_URL, "model": OLLAMA_MODEL}

def ensure_model_available(model: str) -> None:
    """
    Check-only: verifies the model exists in Ollama. Does NOT pull.
    Pulling inside a request handler is brittle and can time out.
    """
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
        tags = r.json().get("models", [])
        print(tags)
        if any((m.get("name") == model) for m in tags):
            return
        raise RuntimeError(
            f"Model '{model}' not found in Ollama. Pull it with: "
            f"docker exec -it ollama ollama pull {model}"
        )
    except Exception as e:
        raise RuntimeError(str(e))

def build_messages(
    user_text: str,
    history: list | None,
    file_note: str | None,
    file_text: str | None
):
    system = {
        "role": "system",
        "content": (
            "You are the backend model for a Conversational UI demo. "
            "Maintain context across turns using the provided chat history. "
            "If a file content is provided, use it as the source of truth. "
            "Be helpful, concise, and respond in plain text. "
            "If unsure, say you're unsure. Don't invent facts. "
            "Your token prediction limit is "+str(NUM_PREDICT)+" so please contain your responses in that to avoid incomplete answers. "
        )
    }

    msgs = [system]

    if history:
        for m in history:
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                msgs.append({"role": role, "content": content})

    content = user_text.strip()

    if file_note:
        content += f"\n\n[Attached file info]\n{file_note}"

    if file_text:
        content += (
            "\n\n[Attached file content]\n"
            "```text\n"
            f"{file_text}\n"
            "```"
        )

    msgs.append({"role": "user", "content": content})
    return msgs

@app.post("/api/chat")
def chat():
    t0 = time.time()

    user_message = ""
    file_note = None
    file_text = None
    history = []

    if request.content_type and request.content_type.startswith("application/json"):
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        history = data.get("history") or []
        if not isinstance(history, list):
            history = []
    else:
        user_message = (request.form.get("message") or "").strip()
        history_raw = request.form.get("history")
        if history_raw:
            try:
                history = json.loads(history_raw)
                if not isinstance(history, list):
                    history = []
            except Exception:
                history = []

        f = request.files.get("file")
        if f and f.filename:
            safe_name = Path(f.filename).name
            ts = int(time.time())
            raw = f.read()
            size_bytes = len(raw)
            if size_bytes > MAX_UPLOAD_MB * 1024 * 1024:
                return jsonify({"error": f"File too large (> {MAX_UPLOAD_MB} MB)."}), 413

            suffix = Path(safe_name).suffix.lower()
            if suffix in TEXT_EXTENSIONS:
                try:
                    file_text = decode_bytes(raw, MAX_FILE_CHARS)
                except Exception as e:
                    file_text = f"[Could not decode text content: {e}]"

            object_name = f"{ts}_{safe_name}"
            if STORAGE_BACKEND == "minio":
                stored = put_object_minio(object_name, raw, content_type=f.mimetype)
                file_note = (
                    f"filename: {safe_name}\n"
                    f"size_bytes: {size_bytes}\n"
                    f"storage: minio\n"
                    f"bucket: {stored['bucket']}\n"
                    f"object: {stored['object']}\n"
                    f"url: {stored['url']}"
                )
            else:
                out_path = UPLOAD_DIR / object_name
                out_path.write_bytes(raw)
                file_note = (
                    f"filename: {safe_name}\n"
                    f"size_bytes: {size_bytes}\n"
                    f"storage: local\n"
                    f"saved_as: {out_path}"
                )

    if not user_message:
        return jsonify({"error": "Missing 'message'."}), 400

    export_fmt = parse_export_intent(user_message)
    if export_fmt:
        src = last_assistant_text(history)
        if not src:
            return jsonify({
                "reply": "I can export, but I don’t have a previous assistant answer in this conversation yet.",
                "model": OLLAMA_MODEL,
                "t_ms": int((time.time() - t0) * 1000),
            })

        try:
            ensure_model_available(OLLAMA_MODEL)
        except Exception as e:
            return jsonify({"error": "Model not available", "details": str(e)}), 503

        try:
            if export_fmt == "txt":
                data_bytes = src.encode("utf-8", errors="replace")
            else:
                csv_text = convert_text_to_csv_via_model(src)
                data_bytes = csv_text.encode("utf-8", errors="replace")

            # store export (prefer MinIO if enabled)
            if STORAGE_BACKEND == "minio":
                file_info = save_export_to_minio(export_fmt, data_bytes)
            else:
                # fallback to local
                ts = int(time.time())
                name = f"export_{ts}.{export_fmt}"
                out_path = UPLOAD_DIR / name
                out_path.write_bytes(data_bytes)
                file_info = {"format": export_fmt, "name": name, "url": f"/static/{name}"}

            reply = f"Done — I exported the last answer as {export_fmt.upper()}."
            return jsonify({
                "reply": reply,
                "model": OLLAMA_MODEL,
                "t_ms": int((time.time() - t0) * 1000),
                "file": {
                    "format": file_info["format"],
                    "name": file_info["name"],
                    "url": file_info["url"],
                },
            })
        except Exception as e:
            return jsonify({"error": "Export failed", "details": str(e)}), 500

    try:
        ensure_model_available(OLLAMA_MODEL)
    except Exception as e:
        return jsonify({"error": "Model not available", "details": str(e)}), 503

    payload = {
        "model": OLLAMA_MODEL,
        "messages": build_messages(user_message, history, file_note, file_text),
        "stream": False,
        "options": {"num_predict": NUM_PREDICT},
    }

    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=(10, 600))
        r.raise_for_status()
        out = r.json()
        reply = (out.get("message") or {}).get("content") or ""
    except requests.HTTPError:
        try:
            err = r.json()
        except Exception:
            err = {"error": r.text}
        return jsonify({"error": "Ollama request failed", "details": err}), 502
    except Exception as e:
        return jsonify({"error": "Backend error", "details": str(e)}), 500

    return jsonify({
        "reply": reply,
        "model": OLLAMA_MODEL,
        "t_ms": int((time.time() - t0) * 1000),
        "file_saved": bool(file_note),
    })

@app.post("/api/chat_stream")
def chat_stream():
    t0 = time.time()

    user_message = ""
    file_note = None
    file_text = None
    history = []

    if request.content_type and request.content_type.startswith("application/json"):
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        history = data.get("history") or []
        if not isinstance(history, list):
            history = []
    else:
        user_message = (request.form.get("message") or "").strip()
        history_raw = request.form.get("history")
        if history_raw:
            try:
                history = json.loads(history_raw)
                if not isinstance(history, list):
                    history = []
            except Exception:
                history = []

        f = request.files.get("file")
        if f and f.filename:
            safe_name = Path(f.filename).name
            ts = int(time.time())
            raw = f.read()
            size_bytes = len(raw)
            if size_bytes > MAX_UPLOAD_MB * 1024 * 1024:
                return jsonify({"error": f"File too large (> {MAX_UPLOAD_MB} MB)."}), 413

            suffix = Path(safe_name).suffix.lower()
            if suffix in TEXT_EXTENSIONS:
                try:
                    file_text = decode_bytes(raw, MAX_FILE_CHARS)
                except Exception as e:
                    file_text = f"[Could not decode text content: {e}]"

            object_name = f"{ts}_{safe_name}"
            if STORAGE_BACKEND == "minio":
                stored = put_object_minio(object_name, raw, content_type=f.mimetype)
                file_note = (
                    f"filename: {safe_name}\n"
                    f"size_bytes: {size_bytes}\n"
                    f"storage: minio\n"
                    f"bucket: {stored['bucket']}\n"
                    f"object: {stored['object']}\n"
                    f"url: {stored['url']}"
                )
            else:
                out_path = UPLOAD_DIR / object_name
                out_path.write_bytes(raw)
                file_note = (
                    f"filename: {safe_name}\n"
                    f"size_bytes: {size_bytes}\n"
                    f"storage: local\n"
                    f"saved_as: {out_path}"
                )

    if not user_message:
        return jsonify({"error": "Missing 'message'."}), 400

    try:
        ensure_model_available(OLLAMA_MODEL)
    except Exception as e:
        return jsonify({"error": "Model not available", "details": str(e)}), 503

    export_fmt = parse_export_intent(user_message)

    @stream_with_context
    def generate():
        yield f"data: {json.dumps({'type':'meta','model':OLLAMA_MODEL,'t0_ms':int((time.time()-t0)*1000)})}\n\n"

        if export_fmt:
            src = last_assistant_text(history)
            if not src:
                yield f"data: {json.dumps({'type':'delta','text':'I can export, but I don’t have a previous assistant answer in this conversation yet.'})}\n\n"
                yield f"data: {json.dumps({'type':'done'})}\n\n"
                return

            try:
                if export_fmt == "txt":
                    data_bytes = src.encode("utf-8", errors="replace")
                else:
                    csv_text = convert_text_to_csv_via_model(src)
                    data_bytes = csv_text.encode("utf-8", errors="replace")

                if STORAGE_BACKEND == "minio":
                    file_info = save_export_to_minio(export_fmt, data_bytes)
                else:
                    ts = int(time.time())
                    name = f"export_{ts}.{export_fmt}"
                    out_path = UPLOAD_DIR / name
                    out_path.write_bytes(data_bytes)
                    file_info = {"format": export_fmt, "name": name, "url": f"/static/{name}"}

                msg = f"Done — I exported the last answer as {export_fmt.upper()}."
                yield f"data: {json.dumps({'type':'delta','text':msg})}\n\n"
                yield f"data: {json.dumps({'type':'file','file':{'format':file_info['format'],'name':file_info['name'],'url':file_info['url']}})}\n\n"
                yield f"data: {json.dumps({'type':'done'})}\n\n"
                return
            except Exception as e:
                yield f"data: {json.dumps({'type':'error','error':'Export failed','details':str(e)})}\n\n"
                yield f"data: {json.dumps({'type':'done'})}\n\n"
                return

        payload = {
            "model": OLLAMA_MODEL,
            "messages": build_messages(user_message, history, file_note, file_text),
            "stream": True,
            "options": {"num_predict": NUM_PREDICT},
        }

        try:
            yield from ollama_chat_stream(payload)
        except requests.HTTPError as e:
            yield f"data: {json.dumps({'type':'error','error':'Ollama request failed','details':str(e)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','error':'Backend error','details':str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")
