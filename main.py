from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from app.recommender import SHLRecommender

app = FastAPI()
recommender = SHLRecommender()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/recommend")
async def recommend(query: str = Query(..., description="Enter job description or skill")):
    results = recommender.recommend(query)
    return results