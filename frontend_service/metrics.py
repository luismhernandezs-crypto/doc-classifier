from prometheus_client import Counter, Histogram
import time

FRONTEND_VISITS = Counter(
    "frontend_visits_total",
    "Total de visitas a la aplicación web"
)

FRONTEND_UPLOADS = Counter(
    "frontend_uploads_total",
    "Cantidad de archivos subidos desde el frontend"
)


FRONTEND_ERRORS = Counter(
    "frontend_errors_total",
    "Errores ocurridos en el frontend"
)
DASHBOARD_RENDER_TIME = Histogram(
    "frontend_dashboard_render_seconds",
    "Tiempo de renderización del dashboard"
)
