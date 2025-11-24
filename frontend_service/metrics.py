from prometheus_client import Counter, Histogram
import time
# Visitas al frontend
FRONTEND_VISITS = Counter(
    "frontend_visits_total",
    "Total de visitas a la aplicación web"
)

# Archivos subidos por los usuarios
FRONTEND_UPLOADS = Counter(
    "frontend_uploads_total",
    "Cantidad de archivos subidos desde el frontend"
)

# Errores del frontend
FRONTEND_ERRORS = Counter(
    "frontend_errors_total",
    "Errores ocurridos en el frontend"
)

# Tiempo que toma renderizar el dashboard
DASHBOARD_RENDER_TIME = Histogram(
    "frontend_dashboard_render_seconds",
    "Tiempo de renderización del dashboard"
)
