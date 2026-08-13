from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="EDKT API Engine")

# 1. Broad CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Vercel frontend, localhost, etc.
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 2. Global Preflight Interceptor (Interprets and satisfies ALL OPTIONS requests before auth checks)
@app.middleware("http")
async def cors_preflight_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        response = Response(status_code=status.HTTP_200_OK)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response
    
    response = await call_next(request)
    return response