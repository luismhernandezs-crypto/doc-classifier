# frontend_service/main.py
from fastapi import FastAPI, UploadFile, File, Form, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from jose import jwt, JWTError
from passlib.context import CryptContext
from datetime import datetime, timedelta
from minio import Minio
import requests, psycopg2, io, os, traceback

# ---------------- CONFIGURACIÓN ----------------
app = FastAPI(title="SMAV OCR & Classifier")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

SECRET_KEY = os.getenv("SECRET_KEY", "super_secret_key_123")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24))
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# NOTA: Estas URLs ya no se usan directamente porque n8n orquesta todo,
# pero las dejamos por si acaso necesitas depurar servicios individuales.
OCR_URL = os.getenv("OCR_URL", "http://ocr_service:8000/extract-text")
CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://classifier_service:8080/classify-text")

# --- CONFIGURACIÓN DE N8N ---
# IMPORTANTE: Dentro de Docker usamos "http://n8n:5678".
# Si el workflow está "Active", usa "/webhook/". Si estás probando con el botón "Execute", usa "/webhook-test/".
# Aquí configuramos la ruta de producción por defecto:
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://n8n:5678/webhook/procesar-documento")

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "doc_classifier"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin123")
}

# MinIO Client (Solo lo usamos aquí para verificar buckets al inicio, 
# n8n se encarga de subir los archivos ahora)
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
        print(f"⚠️ Bucket '{bucket}' no creado: {e}")

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
        print("✅ Base de datos inicializada correctamente.")
    except Exception as e:
        print("⚠️ Error inicializando la BD:", e)

init_db()

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

def create_user(username, password, rol="usuario", email=None):
    conn = get_db_conn()
    cur = conn.cursor()
    hashed = pwd_context.hash(password)
    cur.execute(
        "INSERT INTO usuarios (username, password_hash, rol) VALUES (%s, %s, %s) ON CONFLICT (username) DO NOTHING;",
        (username, hashed, rol),
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
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse("/login", status_code=302)
    try:
        payload = decode_token(token)
        user = {"username": payload.get("sub"), "rol": payload.get("rol")}
        return templates.TemplateResponse("index.html", {"request": request, "user": user})
    except JWTError:
        return RedirectResponse("/login", status_code=302)

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login_post(username: str = Form(...), password: str = Form(...)):
    user = get_user(username)
    if not user or not pwd_context.verify(password, user["password_hash"]):
        return HTMLResponse("<p>Usuario o contraseña inválidos.</p>", status_code=400)
    token = create_token(username, user["rol"])
    resp = RedirectResponse("/", status_code=302)
    resp.set_cookie("access_token", token, httponly=True, samesite="lax")
    return resp

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
def register_post(username: str = Form(...), password: str = Form(...), email: str = Form(None)):
    if get_user(username):
        return HTMLResponse("<p>El usuario ya existe.</p>", status_code=400)
    create_user(username, password)
    return RedirectResponse("/login", status_code=302)

@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie("access_token")
    return resp

# ---------- INTEGRACIÓN CON N8N (NUEVO FLUJO) ----------
@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request, user: dict = Depends(get_current_user_by_request)):
    return templates.TemplateResponse("upload.html", {"request": request, "user": user, "text": None, "category": None})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...), user: dict = Depends(get_current_user_by_request)):
    try:
        # 1. Leer el archivo
        file_bytes = await file.read()
        
        # 2. Enviar a n8n (WebHook)
        # Usamos 'data' como key porque así se configuró en el Webhook de n8n
        files = {'data': (file.filename, file_bytes, file.content_type)}
        
        print(f"🚀 Enviando archivo {file.filename} a n8n ({N8N_WEBHOOK_URL})...")
        
        # Enviamos la petición POST a n8n
        response = requests.post(N8N_WEBHOOK_URL, files=files, timeout=600)
        
        if not response.ok:
            return HTMLResponse(f"<p>Error en n8n: {response.status_code} - {response.text}</p>", status_code=500)
        
        # 3. Procesar respuesta JSON de n8n
        n8n_result = response.json()
        
        # Extraemos los datos que nos devuelve el nodo 'Respond to Webhook'
        status = n8n_result.get("status", "unknown")
        categoria = n8n_result.get("final_category", "Desconocido")
        confianza_smav = n8n_result.get("smav_confidence", "N/A")
        mensaje = n8n_result.get("message", "")
        archivo_txt = n8n_result.get("classified_file_minio", "")

        # Construimos un texto de resumen para guardar en la BD local y mostrar en pantalla
        resumen_proceso = (
            f"✅ Procesado por n8n.\n"
            f"Categoría: {categoria}\n"
            f"Confianza SMAV: {confianza_smav}\n"
            f"Archivo TXT: {archivo_txt}\n"
            f"Mensaje: {mensaje}"
        )

        # 4. Guardar Historial en PostgreSQL (Frontend DB)
        # Aunque n8n guarda archivos, nosotros guardamos el registro de quién lo subió
        try:
            conn = get_db_conn()
            cur = conn.cursor()
            cur.execute("INSERT INTO clasificaciones (texto, categoria, fecha, username) VALUES (%s,%s,%s,%s)",
                        (resumen_proceso, categoria, datetime.now(), user["username"]))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print("⚠️ Error guardando en BD local:", e)

        # 5. Mostrar resultado al usuario
        return templates.TemplateResponse("upload.html", {
            "request": request,
            "user": user,
            "result": resumen_proceso,
            "category": categoria
        })

    except Exception as e:
        print("Upload flow error:", traceback.format_exc())
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
        print("⚠️ history error:", e)
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
        print("⚠️ admin_panel error:", e)
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
        print("⚠️ admin_stats error:", e)
    return JSONResponse(stats)

# ---------- Arranque ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
