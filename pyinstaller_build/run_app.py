"""PyInstaller entry point — imports the FastAPI app object directly rather
than uvicorn's "module:app" import-string form, so the frozen build doesn't
need uvicorn's dynamic module resolution to find image_insight.api.main.
"""
import uvicorn

from image_insight.api.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
