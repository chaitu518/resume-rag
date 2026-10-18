"""Resume RAG Job Matcher - web app entry point.

    python app.py            # http://localhost:8000  (UI)  /docs (Swagger)
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from api.endpoints import router

app = FastAPI(title="Resume RAG Job Matcher", version="1.0")
app.include_router(router, prefix="/api")
UI = Path(__file__).parent / "ui" / "index.html"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(UI)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
