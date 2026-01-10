from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn
from app.core.report_summarizer_api import router as police_report_router


def get_application() -> FastAPI:
    application = FastAPI(
        title="Tasco Insurance Claims Automation API",
        docs_url="/swaggers/",
        redoc_url='/re-docs',
        openapi_url=f"/api/v1/openapi.json",
        description='''
        Tasco Insurance Claims Automation API
            - Preprocessing Police Reports
            - Extracting Police Information and Summarizing using Claude 4 Sonnet
            - Generating Police Summary Reports
            - Performing Specialist Insurance Claims Analysis and Diagnosis
            - Generating Insurance Claims Reports
        '''
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    BASE_DIR = Path(__file__).resolve().parent  # => /code/app

    application.mount(
        "/static",
        StaticFiles(directory=str(BASE_DIR / "static")),  # -> /code/app/static
        name="static"
    )
    
    # Include API routers
    application.include_router(police_report_router)
    return application

app = get_application()

if __name__ == '__main__':
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")