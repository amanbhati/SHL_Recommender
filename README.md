# SHL Assessment Recommendation System

🧠 Uses NLP and semantic similarity to match job descriptions to SHL assessments.

## How to Run

Step 1: Install dependencies
pip install -r requirements.txt

Step 2: Run FastAPI backend
uvicorn app.main:app --reload

Step 3: Run Streamlit frontend
streamlit run frontend/streamlit_app.py