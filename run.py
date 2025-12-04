# run.py
from waitress import serve
from app import app  # your Flask app instance

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5500)
