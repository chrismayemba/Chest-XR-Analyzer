from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import numpy as np
from models.swin_transformer import XRayProcessor

app = FastAPI(title="X-ray Analyzer")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the X-ray processor
xray_processor = XRayProcessor()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/analyze/")
async def analyze_xray(file: UploadFile = File(...)):
    # Verify file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Get analysis results
        results = xray_processor.analyze(image)
        
        # Convert numpy arrays to lists for JSON serialization
        results['segmentation'] = results['segmentation'].tolist()
        results['attention'] = results['attention'].tolist()
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
