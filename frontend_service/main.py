# frontend_service/main.py (CORREGIDO)
from fastapi import FastAPI, UploadFile, File, Form, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from jose import jwt, JWTError
from passlib.context import CryptContext
from datetime import datetime, timedelta
from minio import Minio
from prometheus_client import make_asgi_app
import requests, psycopg2, io, os, traceback, time, logging, re
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
import json

# métricas definidas en frontend_service/metrics.py
from metrics import (
    FRONTEND_VISITS,
    FRONTEND_UPLOADS,
    FRONTEND_ERRORS,
    DASHBOARD_RENDER_TIME,
)

# ---------------- CONFIGURACIÓN ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("frontend")

app = FastAPI(title="SMAV OCR & Classifier")
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

SECRET_KEY = os.getenv("SECRET_KEY", "super_secret_key_123")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24))
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

OCR_URL = os.getenv("OCR_URL", "http://ocr_service:8000/extract-text")
CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://classifier_service:8080/classify-text")
# Default for container-to-container requests inside Docker network:
SMAV_METRICS_URL = os.getenv("SMAV_METRICS_URL", "http://smav_service:8000/metrics")
SMAV_PROCESS_URL = os.getenv("SMAV_PROCESS_URL", "http://smav_service:8000/process-document")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://n8n:5678/webhook/procesar-documento")

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "doc_classifier"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin123")
}

MINIO_CLIENT = Minio(
    os.getenv("MINIO_HOST", "minio:9000"),
    access_key=os.getenv("MINIO_ROOT_USER", "admin"),
    secret_key=os.getenv("MINIO_ROOT_PASSWORD", "admin123"),
    secure=False
)

for bucket in ["incoming-docs", "classified-docs"]:
    try:
        if not MINIO_CLIENT.bucket_exists(bucket):
            MINIO_CLIENT.make_bucket(bucket)
    except Exception as e:
        logger.warning(f"Bucket '{bucket}' no creado: {e}")

# ---------------- DB ----------------
def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)

def init_db():
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS usuarios (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                rol VARCHAR(20) DEFAULT 'usuario' CHECK (rol IN ('usuario','admin'))
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS clasificaciones (
                id SERIAL PRIMARY KEY,
                texto TEXT,
                categoria VARCHAR(255),
                fecha TIMESTAMP DEFAULT NOW(),
                username VARCHAR(100) REFERENCES usuarios(username)
            );
        """)
        cur.execute("""
            INSERT INTO usuarios (username, password_hash, rol)
            VALUES ('admin', '$2b$12$AFQe9b6uohAzbct6mGv5ueZ.zvayKv7QMLm8Y9PfAGHHqCwPZB42K', 'admin')
            ON CONFLICT (username) DO NOTHING;
        """)
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Base de datos inicializada correctamente.")
    except Exception as e:
        logger.exception("Error inicializando la BD: %s", e)

# Retry init_db a few times (Postgres container might not be ready)
for _ in range(8):
    try:
        init_db()
        break
    except Exception:
        time.sleep(2)

# ---------------- AUTH ----------------
def create_token(username, rol):
    exp = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "rol": rol, "exp": exp}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")

def get_user(username):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT username, password_hash, rol FROM usuarios WHERE username=%s", (username,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return {"username": row[0], "password_hash": row[1], "rol": row[2]}
    return None

def create_user(username, password, rol="usuario"):
    conn = get_db_conn()
    cur = conn.cursor()
    hashed = pwd_context.hash(password)
    cur.execute(
        "INSERT INTO usuarios (username, password_hash, rol) VALUES (%s, %s, %s) ON CONFLICT (username) DO NOTHING;",
        (username, hashed, rol)
    )
    conn.commit()
    cur.close()
    conn.close()

def get_current_user_by_request(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="No autenticado")
    payload = decode_token(token)
    return {"username": payload.get("sub"), "rol": payload.get("rol")}

# ---------------- RUTAS ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    start = time.time()
    FRONTEND_VISITS.inc()

    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse("/login", status_code=302)
    try:
        payload = decode_token(token)
        user = {"username": payload.get("sub"), "rol": payload.get("rol")}
        DASHBOARD_RENDER_TIME.observe(time.time() - start)
        return templates.TemplateResponse("index.html", {"request": request, "user": user})
    except JWTError:
        FRONTEND_ERRORS.inc()
        return RedirectResponse("/login", status_code=302)

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    FRONTEND_VISITS.inc()
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login_post(username: str = Form(...), password: str = Form(...)):
    start = time.time()
    FRONTEND_VISITS.inc()
    try:
        user = get_user(username)
        if not user or not pwd_context.verify(password, user["password_hash"]):
            FRONTEND_ERRORS.inc()
            return HTMLResponse("<p>Usuario o contraseña inválidos.</p>", status_code=400)

        token = create_token(username, user["rol"])
        resp = RedirectResponse("/", status_code=302)
        resp.set_cookie(
            "access_token",
            token,
            httponly=True,
            samesite="lax",
            secure=False
        )
        DASHBOARD_RENDER_TIME.observe(time.time() - start)
        return resp
    except Exception as e:
        FRONTEND_ERRORS.inc()
        logger.exception("login_post error: %s", e)
        return HTMLResponse("<p>Error interno.</p>", status_code=500)

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    FRONTEND_VISITS.inc()
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
def register_post(username: str = Form(...), password: str = Form(...)):
    start = time.time()
    FRONTEND_VISITS.inc()
    try:
        if get_user(username):
            FRONTEND_ERRORS.inc()
            return HTMLResponse("<p>El usuario ya existe.</p>", status_code=400)
        create_user(username, password)
        DASHBOARD_RENDER_TIME.observe(time.time() - start)
        return RedirectResponse("/login", status_code=302)
    except Exception as e:
        FRONTEND_ERRORS.inc()
        logger.exception("register_post error: %s", e)
        return HTMLResponse("<p>Error registrando usuario.</p>", status_code=500)

@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie("access_token", path="/")
    return resp

# ---------- Predict ----------
@app.post("/predict")
def predict_text(text: str = Form(...), user: dict = Depends(get_current_user_by_request)):
    try:
        resp = requests.post(CLASSIFIER_URL, json={"text": text}, timeout=10)
        if not resp.ok:
            FRONTEND_ERRORS.inc()
            return JSONResponse({"error": "Error clasificando texto"}, status_code=500)
        result = resp.json()
        return {
            "categoria": result.get("categoria", "Desconocido"),
            "confianza": result.get("confianza", 0)
        }
    except Exception as e:
        FRONTEND_ERRORS.inc()
        logger.exception("predict error: %s", e)
        return JSONResponse({"error": "Error interno clasificando"}, status_code=500)

# ---------- Upload & SMAV Integration ----------
@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request, user: dict = Depends(get_current_user_by_request)):
    return templates.TemplateResponse("upload.html", {"request": request, "user": user, "text": None, "category": None})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...), user: dict = Depends(get_current_user_by_request)):
    start = time.time()
    FRONTEND_UPLOADS.inc()
    try:
        file_bytes = await file.read()
        files = {'file': (file.filename, file_bytes, file.content_type)}
        response = requests.post(SMAV_PROCESS_URL, files=files, timeout=180)
        if not response.ok:
            FRONTEND_ERRORS.inc()
            return HTMLResponse(f"<p>Error en SMAV: {response.status_code} - {response.text}</p>", status_code=500)
        smav_result = response.json()
        texto_ocr = smav_result.get("extracted_text", "")
        smav_categoria = smav_result.get("final_category", "Desconocido")
        confianza_smav = smav_result.get("smav_confidence", None)
        archivo_txt = smav_result.get("classified_file_minio", "")
        resumen_proceso = (
            f"Procesado por SMAV.\n"
            f"Categoría: {smav_categoria}\n"
            f"Confianza SMAV: {confianza_smav}\n"
            f"Archivo TXT: {archivo_txt}\n"
            f"Mensaje: {smav_result.get('status','')}"
        )

        # --- CALL classifier_service to obtain classifier_pred (we'll use it as ground_truth) ---
        classifier_pred = None
        try:
            if texto_ocr:
                resp = requests.post(CLASSIFIER_URL, json={"text": texto_ocr}, timeout=10)
                if resp.ok:
                    j = resp.json()
                    # Ajusta la clave si tu classifier devuelve otra (categoria, label, etc.)
                    classifier_pred = j.get("categoria") or j.get("category") or j.get("label") or None
                else:
                    logger.warning(f"Classifier returned non-OK: {resp.status_code}")
        except Exception as e:
            logger.exception("Error calling classifier service: %s", e)
            classifier_pred = None

        # Define ground_truth: usamos classifier_pred (puedes cambiar la lógica si quieres)
        ground_truth = classifier_pred

        # Guardar en DB con campos extendidos (smav_pred, classifier_pred, ground_truth, confidence)
        try:
            conn = get_db_conn()
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS clasificaciones (
                    id SERIAL PRIMARY KEY,
                    texto TEXT,
                    categoria VARCHAR(255),
                    fecha TIMESTAMP DEFAULT NOW(),
                    username VARCHAR(100),
                    smav_pred VARCHAR(255),
                    classifier_pred VARCHAR(255),
                    ground_truth VARCHAR(255),
                    confidence NUMERIC
                );
            """)
            cur.execute(
                """
                INSERT INTO clasificaciones
                (texto, categoria, fecha, username, smav_pred, classifier_pred, ground_truth, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (resumen_proceso, smav_categoria, datetime.now(), user["username"], smav_categoria, classifier_pred, ground_truth, confianza_smav)
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.exception("Error guardando en BD local: %s", e)

        DASHBOARD_RENDER_TIME.observe(time.time() - start)
        return templates.TemplateResponse("upload.html", {
            "request": request,
            "user": user,
            "result": resumen_proceso,
            "category": smav_categoria,
            "texto_ocr": texto_ocr
        })
    except Exception as e:
        FRONTEND_ERRORS.inc()
        logger.exception("Upload flow error: %s", e)
        return HTMLResponse(f"<p>Error procesando el archivo: {e}</p>", status_code=500)

# ---------- HISTORIAL ----------
@app.get("/history", response_class=HTMLResponse)
def history(request: Request, user: dict = Depends(get_current_user_by_request)):
    rows = []
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        if user["rol"] == "admin":
            cur.execute("SELECT texto, categoria, fecha, username FROM clasificaciones ORDER BY fecha DESC")
        else:
            cur.execute("SELECT texto, categoria, fecha, username FROM clasificaciones WHERE username=%s ORDER BY fecha DESC", (user["username"],))
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        FRONTEND_ERRORS.inc()
        logger.exception("history error: %s", e)
    return templates.TemplateResponse("history.html", {"request": request, "rows": rows, "user": user})

# ---------- ADMIN ----------
@app.get("/admin", response_class=HTMLResponse)
def admin_panel(request: Request, user: dict = Depends(get_current_user_by_request)):
    if user["rol"] != "admin":
        raise HTTPException(status_code=403, detail="No autorizado")
    users, stats = [], []
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT username, rol FROM usuarios ORDER BY username;")
        users = cur.fetchall()
        cur.execute("SELECT username, COUNT(*) FROM clasificaciones GROUP BY username;")
        stats = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        FRONTEND_ERRORS.inc()
        logger.exception("admin_panel error: %s", e)
    return templates.TemplateResponse("admin.html", {"request": request, "users": users, "stats": stats, "user": user})

@app.get("/admin/stats")
def admin_stats(user: dict = Depends(get_current_user_by_request)):
    if user["rol"] != "admin":
        raise HTTPException(status_code=403, detail="No autorizado")
    stats = []
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT username, COUNT(*) FROM clasificaciones GROUP BY username;")
        stats = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        FRONTEND_ERRORS.inc()
        logger.exception("admin_stats error: %s", e)
    return JSONResponse(stats)

# ---------- MÉTRICAS GLOBALES (SOLO ADMIN) ----------
@app.get("/admin/metrics", response_class=HTMLResponse)
def admin_metrics(request: Request, user: dict = Depends(get_current_user_by_request)):
    FRONTEND_VISITS.inc()

    if user["rol"] != "admin":
        raise HTTPException(status_code=403, detail="No autorizado")

    try:
        conn = get_db_conn()
        cur = conn.cursor()

        # Total documentos clasificados
        cur.execute("SELECT COUNT(*) FROM clasificaciones;")
        total_docs = cur.fetchone()[0]

        # Clasificados por categoría
        cur.execute("""
            SELECT categoria, COUNT(*)
            FROM clasificaciones
            GROUP BY categoria
            ORDER BY COUNT(*) DESC;
        """)
        por_categoria = [{"categoria": r[0], "cantidad": r[1]} for r in cur.fetchall()]

        # Clasificados por usuario
        cur.execute("""
            SELECT username, COUNT(*)
            FROM clasificaciones
            GROUP BY username
            ORDER BY COUNT(*) DESC;
        """)
        por_usuario = [{"username": r[0], "cantidad": r[1]} for r in cur.fetchall()]

        # Categoría más usada
        cur.execute("""
            SELECT categoria, COUNT(*) AS cnt
            FROM clasificaciones
            GROUP BY categoria
            ORDER BY cnt DESC LIMIT 1;
        """)
        top_categoria_row = cur.fetchone()
        top_categoria = {
            "categoria": top_categoria_row[0],
            "cantidad": top_categoria_row[1]
        } if top_categoria_row else None

        cur.close()
        conn.close()

        # -------- LEER METRICAS DE SMAV --------
        prometheus_raw = ""
        prometheus_error = None
        try:
            resp = requests.get(SMAV_METRICS_URL, timeout=10)
            if resp.status_code == 200:
                prometheus_raw = resp.text
            else:
                prometheus_error = f"SMAV returned status {resp.status_code}"
        except Exception as e:
            prometheus_error = f"ERROR obteniendo métricas de SMAV: {e}"

        # -------- PARSEADOR ROBUSTO --------
        metrics_list = []

        if prometheus_raw:
            help_map = {}
            type_map = {}

            for line in prometheus_raw.splitlines():
                line = line.strip()

                # HELP
                if line.startswith("# HELP"):
                    parts = line.split(" ", 3)
                    if len(parts) >= 4:
                        metric_name = parts[2]
                        help_map[metric_name] = parts[3]

                # TYPE
                elif line.startswith("# TYPE"):
                    parts = line.split(" ", 3)
                    if len(parts) >= 4:
                        metric_name = parts[2]
                        type_map[metric_name] = parts[3]

                # Métrica real
                elif line and not line.startswith("#"):
                    match = re.match(
                        r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*?\})?\s+([0-9\.\+eE-]+)$',
                        line
                    )
                    if match:
                        metric_name = match.group(1)
                        labels = match.group(2) or ""
                        value = match.group(3)

                        metrics_list.append({
                            "name": metric_name,
                            "labels": labels,
                            "value": value,
                            "help": help_map.get(metric_name, ""),
                            "type": type_map.get(metric_name, "")
                        })

        # -------- RESPUESTA --------
        return templates.TemplateResponse("metrics.html", {
            "request": request,
            "user": user,
            "total_docs": total_docs,
            "por_categoria": por_categoria,
            "por_usuario": por_usuario,
            "top_categoria": top_categoria,
            "prometheus_raw": prometheus_raw,
            "prometheus_error": prometheus_error,
            "metrics_list": metrics_list
        })

    except Exception as e:
        FRONTEND_ERRORS.inc()
        logger.exception("admin_metrics error: %s", e)
        return HTMLResponse("<p>Error obteniendo métricas</p>", status_code=500)

# ---------- MÉTRICAS DE DOCUMENTOS SUBIDOS ----------
@app.get("/admin/metrics/uploaded-documents", response_class=HTMLResponse)
def metrics_uploaded_documents(request: Request, user: dict = Depends(get_current_user_by_request)):
    if user["rol"] != "admin":
        raise HTTPException(status_code=403, detail="No autorizado")
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        # Traemos las columnas que necesitamos
        cur.execute("SELECT smav_pred, ground_truth FROM clasificaciones;")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return HTMLResponse("<p>No hay registros en la base de datos para calcular métricas.</p>", status_code=400)

        true_labels = []
        predicted_labels = []

        for r in rows:
            smav_pred = r[0] or None
            ground = r[1] or None
            # Añadimos sólo pares válidos
            if ground and smav_pred:
                true_labels.append(ground)
                predicted_labels.append(smav_pred)

        if not true_labels:
            return HTMLResponse("<p>No hay pares ground-truth / smav_pred válidos para calcular métricas.</p>", status_code=400)

        # Calcular métricas
        all_labels = sorted(list(set(true_labels) | set(predicted_labels)))
        precision, recall, fscore, support = precision_recall_fscore_support(
            true_labels, predicted_labels, labels=all_labels, zero_division=0
        )
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1_macro = fscore.mean()

        # Matriz de confusión
        cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(len(all_labels)))
        ax.set_yticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=45, ha='right')
        ax.set_yticklabels(all_labels)
        plt.colorbar(im)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        metrics_output = {
            "labels": all_labels,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "fscore": fscore.tolist(),
            "support": support.tolist(),
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "confusion_matrix_img": img_base64
        }

        return templates.TemplateResponse("metrics.html", {
            "request": request,
            "user": user,
            "metrics": metrics_output
        })
    except Exception as e:
        FRONTEND_ERRORS.inc()
        logger.exception("Error calculando métricas: %s", e)
        return HTMLResponse("<p>Error al calcular las métricas.</p>", status_code=500)

# ---------- Arranque ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
