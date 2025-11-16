import numpy as np
from typing import List, Tuple

class HybridScorer:
    def __init__(self, texts: List[str], vector_embeddings: List[List[float]], bm25_weight: float = 0.3, vector_weight: float = 0.7):
        if not (0 <= bm25_weight <= 1 and 0 <= vector_weight <= 1 and (bm25_weight + vector_weight) > 0):
            raise ValueError("가중치는 0과 1 사이여야 하며, 합이 0보다 커야 합니다.")
        
        self.texts = texts
        self.vector_embeddings = np.array(vector_embeddings)
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-Max 정규화를 사용하여 점수를 0과 1 사이로 조정합니다."""
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score == min_score:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def search(self, query_bm25: List[float], query_vector: List[float]) -> List[Tuple[str, float]]:
        """하이브리드 검색을 수행하고 (문서, 점수) 튜플 리스트를 반환합니다."""
        # 정규화
        query_bm25 = np.array(query_bm25)
        normalized_bm25_scores = self._normalize_scores(query_bm25)

        
        # 코사인 유사도 계산: (A · B) / (||A|| * ||B||)
        query_vec = np.array(query_vector)
        dot_products = np.dot(self.vector_embeddings, query_vec)
        norm_query = np.linalg.norm(query_vec)
        norm_docs = np.linalg.norm(self.vector_embeddings, axis=1)
        
        valid_indices = (norm_query > 0) & (norm_docs > 0)
        cosine_sim = np.zeros(len(self.texts))
        cosine_sim[valid_indices] = dot_products[valid_indices] / (norm_query * norm_docs[valid_indices])
        
        normalized_vector_scores = self._normalize_scores(cosine_sim)

        # 3. 가중치 합산
        hybrid_scores = (self.bm25_weight * normalized_bm25_scores) + (self.vector_weight * normalized_vector_scores)

        # --- 계산 과정 출력 코드 추가 ---
        print("\n--- 각 문서별 하이브리드 점수 계산 과정 ---")
        for i in range(len(self.texts)):
            bm25_tot = self.bm25_weight * normalized_bm25_scores[i]
            vec_tot = self.vector_weight * normalized_vector_scores[i]
            print(f"문서 {i+1}: \"{self.texts[i][:25]}...\"")
            print(f"  - 최종 점수: {hybrid_scores[i]:.4f} = (BM25 가중 점수: {bm25_tot:.4f}) + (Vector 가중 점수: {vec_tot:.4f})")
        # --------------------------------
        
        # 점수가 높은 순으로 정렬
        sorted_results = sorted(zip(self.texts, hybrid_scores), key=lambda item: item[1], reverse=True)
        return sorted_results