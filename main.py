import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import io
import os
import joblib
from databricks.sdk import WorkspaceClient

# Configuración de la APP
app = FastAPI(title="API Predicción Incendios")
templates = Jinja2Templates(directory="pages")

# --- DEFINICIÓN DE DATOS (INPUT) ---
class IncendioInput(BaseModel):
    claseincendio: str
    numeromediospersonal: int
    latitud: float
    longitud: float
    altitud: float
    anio: int
    velocidadviento: float
    numeromediospesados: int
    humrelativa: float
    tempmaxima: float
    diasultimalluvia: float
    idprovincia: int
    probabilidadignicion: float
    idcomunidad: int
    numeromediosaereos: int
    iddetectadopor: str
    idpeligro: float
    tipodeataque: str
    combustible: str
    horadeteccion: str

# --- CONFIGURACIÓN DATABRICKS ---
# NOTA: Estas variables se configurarán en Render, no aquí.
DB_HOST = os.getenv("DATABRICKS_HOST")
DB_TOKEN = os.getenv("DATABRICKS_TOKEN")
ruta_volumen = "/Volumes/workspace/default/prediccion_incendios/modelo_incendios_rf.pkl"

# Variable global para el modelo
model = None
model_status = "cargando"

# --- CARGA DEL MODELO AL INICIAR ---
# Es mejor cargar el modelo al inicio o bajo demanda controlada
try:
    if not DB_HOST or not DB_TOKEN:
        raise ValueError("Faltan las credenciales de Databricks en las variables de entorno.")

    w = WorkspaceClient(host=DB_HOST, token=DB_TOKEN)

    print("Accediendo al modelo en Databricks...")
    response = w.files.download(ruta_volumen)
    
    model_bytes = response.contents.read()
    
    if not model_bytes:
        raise ValueError("El archivo descargado está vacío.")

    buffer = io.BytesIO(model_bytes)
    model = joblib.load(buffer)
    
    model_status = "ok"
    print("¡Modelo cargado exitosamente!")

except Exception as e:
    print(f"Error cargando el modelo: {e}")
    model_status = "ko"

# --- ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    # Aseguramos que el HTML sepa si el modelo está listo
    return templates.TemplateResponse("index.html", {"request": request, "status": model_status})

@app.get("/health")
def health():
    return {"status": model_status}

@app.post("/predict")
def predict(input_data: IncendioInput):
    if model_status != "ok" or model is None:
        raise HTTPException(status_code=500, detail="El modelo no se ha cargado correctamente desde Databricks.")
    
    # Convertir a DataFrame (Pydantic v2 usa model_dump, v1 usa dict. Usamos dict por compatibilidad)
    try:
        df_input = pd.DataFrame([input_data.dict()])
        
        # Realizar predicción
        prediction = model.predict(df_input)
        
        # Ajustar la salida según tu lógica de negocio
        resultado = int(prediction[0])
        
        return {
            "prediccion": resultado,
            "mensaje": "Incendio Peligroso" if resultado == 1 else "Incendio Controlable" # Ajusta este texto según lo que signifique 0 y 1
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la predicción: {str(e)}")
    