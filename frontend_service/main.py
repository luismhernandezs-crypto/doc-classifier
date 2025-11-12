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
app = FastAPI(title="OCR + Classifier + MinIO + Login")

# monta static (asegúrate de que ./static existe en tu repo; puede estar vacío)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

SECRET_KEY = os.getenv("SECRET_KEY", "super_secret_key_123")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24))
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

OCR_URL = os.getenv("OCR_URL", "http://ocr_service:8000/extract-text")
CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://classifier_service:8080/classify-text")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://n8n:5678/webhook/new_user")

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

# intentamos crear buckets si es posible (no romper si MinIO no está listo)
for bucket in ["incoming-docs", "classified-docs"]:
    try:
        if not MINIO_CLIENT.bucket_exists(bucket):
            MINIO_CLIENT.make_bucket(bucket)
    except Exception as e:
        print(f"⚠️ Verificación/creación bucket '{bucket}' falló: {e}")

# ---------------- DB helpers ----------------
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
        # usuario admin por defecto (hash para 'admin123' ya incluido)
        cur.execute("""
            INSERT INTO usuarios (username, password_hash, rol)
            VALUES ('admin', '$2b$12$AFQe9b6uohAzbct6mGv5ueZ.zvayKv7QMLm8Y9PfAGHHqCwPZB42K', 'admin')
            ON CONFLICT (username) DO NOTHING;
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Tablas verificadas e inicializadas correctamente.")
    except Exception as e:
        print("⚠️ init_db: no se pudo inicializar la BD (espera que postgres arranque). Error:", e)

# Llamada segura a init_db (si postgres no está listo, no mata la app)
init_db()

# ---------------- auth helpers ----------------
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
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT username, password_hash, rol FROM usuarios WHERE username=%s", (username,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return {"username": row[0], "password_hash": row[1], "rol": row[2]}
    except Exception as e:
        print("⚠️ get_user error:", e)
    return None

def create_user(username, password, rol="usuario", email=None):
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        hashed = pwd_context.hash(password)
        cur.execute("INSERT INTO usuarios (username, password_hash, rol) VALUES (%s, %s, %s) ON CONFLICT (username) DO NOTHING;",
                    (username, hashed, rol))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("⚠️ create_user error:", e)
        raise

def get_current_user_by_request(request: Request):
    """
    Intención: obtener usuario desde la cookie (usado en dependencias).
    Lanzará HTTPException(401) si no hay cookie o token inválido.
    """
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="No autenticado")
    payload = decode_token(token)
    return {"username": payload.get("sub"), "rol": payload.get("rol")}

# ---------------- RUTAS ----------------

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    # muestra el formulario de login
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
    # notificar a n8n (no bloqueante)
    try:
        requests.post(N8N_WEBHOOK_URL, json={"username": username, "email": email}, timeout=3)
    except Exception as e:
        print("⚠️ Error notificando a n8n:", e)
    return RedirectResponse("/login", status_code=302)

@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie("access_token")
    return resp

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Ruta pública que redirige al login si no hay sesión válida;
    si el token es válido, renderiza dashboard.
    (No lanzamos 401 desde aquí para evitar errores HTTP inesperados al abrir la raíz).
    """
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse("/login", status_code=302)
    try:
        payload = decode_token(token)
        user = {"username": payload.get("sub"), "rol": payload.get("rol")}
        # show_upload False por defecto para usuarios normales (el dashboard contiene JS para mostrar)
        return templates.TemplateResponse("dashboard.html", {"request": request, "user": user, "show_upload": False})
    except Exception:
        # token inválido/expirado → redirigir a login (elimina cookie en cliente la próxima vez)
        return RedirectResponse("/login", status_code=302)

@app.get("/upload-form", response_class=HTMLResponse)
def upload_form(request: Request, user: dict = Depends(get_current_user_by_request)):
    """
    Mostrar dashboard con el formulario visible (ruta usada por la sidebar 'Subir archivo').
    """
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user, "show_upload": True})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...), user: dict = Depends(get_current_user_by_request)):
    """
    Recibe archivo, llama OCR y classifier, guarda en MinIO y en la BD
    y reusa dashboard.html para mostrar resultado.
    """
    try:
        file_bytes = await file.read()
        files = {"file": (file.filename, file_bytes, file.content_type)}

        # OCR (intentamos, si falla seguimos)
        try:
            ocr_resp = requests.post(OCR_URL, files=files, timeout=12)
            extracted_text = ocr_resp.json().get("extracted_text", "")
        except Exception as e:
            print("⚠️ OCR error:", e)
            extracted_text = ""

        # CLASSIFIER (intentamos)
        try:
            cls_resp = requests.post(CLASSIFIER_URL, json={"text": extracted_text}, timeout=8)
            classifier_json = cls_resp.json() if cls_resp.ok else {}
            category = classifier_json.get("categoria") or classifier_json.get("category") or "Desconocido"
        except Exception as e:
            print("⚠️ CLASSIFIER error:", e)
            category = "Desconocido"

        # Guardar en MinIO (no interrumpimos si falla)
        try:
            MINIO_CLIENT.put_object("incoming-docs", f"{user['username']}/{file.filename}",
                                    io.BytesIO(file_bytes), length=len(file_bytes), content_type=file.content_type)
            text_bytes = extracted_text.encode("utf-8")
            classified_name = f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            MINIO_CLIENT.put_object("classified-docs", f"{user['username']}/{classified_name}",
                                    io.BytesIO(text_bytes), length=len(text_bytes), content_type="text/plain")
        except Exception as e:
            print("⚠️ MinIO put_object error:", e)

        # Guardar registro en BD (no interrumpimos la vista si falla)
        try:
            conn = get_db_conn()
            cur = conn.cursor()
            cur.execute("INSERT INTO clasificaciones (texto, categoria, fecha, username) VALUES (%s,%s,%s,%s)",
                        (extracted_text[:500], category, datetime.now(), user["username"]))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print("⚠️ DB insert clasificacion error:", e)

        # Reusar dashboard para mostrar resultado (evita depender de result.html)
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "user": user,
            "show_upload": True,
            "text": extracted_text,
            "category": category
        })
    except Exception as e:
        print("Upload flow error:", traceback.format_exc())
        return HTMLResponse(f"<p>Error procesando el archivo: {e}</p>", status_code=500)

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

@app.get("/admin", response_class=HTMLResponse)
def admin_panel(request: Request, user: dict = Depends(get_current_user_by_request)):
    if user["rol"] != "admin":
        raise HTTPException(status_code=403, detail="No autorizado")
    users = []
    stats = []
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

# Si arrancas directamente (no necesario dentro de Docker Compose)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
