from sentence_transformers import SentenceTransformer, util
import json
import torch

class SHLRecommender:
    def __init__(self, catalog_path='app/assessments.json'):
        with open(catalog_path, 'r') as f:
            data = json.load(f)
        self.assessments = data['assessments']
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode([a['name'] for a in self.assessments], convert_to_tensor=True)

    def recommend(self, query, top_k=10):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        top_results = torch.topk(scores, k=min(top_k, len(self.assessments)))
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            assessment = self.assessments[int(idx)]
            results.append({**assessment, "score": round(float(score), 2)})
        return results