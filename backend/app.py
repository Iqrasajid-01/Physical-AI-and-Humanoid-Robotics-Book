# Hugging Face Space entry point
# This file allows Hugging Face Spaces to properly detect and run the FastAPI app

import os
import sys
from pathlib import Path

# Add the parent directory to Python path so relative imports in src work correctly
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Change working directory to backend directory to ensure relative imports work
os.chdir(backend_dir)

# Now import the app from src
from src.app import app

# This allows Hugging Face Spaces to properly import and run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))