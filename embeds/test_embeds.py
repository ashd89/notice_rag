import sys
import os
import asyncio
import numpy as np

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from embeds.embed import embed_texts_by_openai, embed_text_by_bm25
from embeds.hybrid_scorer import HybridScorer
from embeds.embed import tokenize_before_bm25

async def main():
    print("=== 하이브리드 검색 테스트 ===")
    
    corpus = [
        "상어는 바다의 포식자입니다.",
        "고래는 매우 큰 포유류입니다.",
        "기린은 목이 긴 동물로, 주로 아프리카 초원에 서식합니다.",
        "상어와 고래는 모두 바다에 사는 생물이지만, 먹이 사슬에서 다른 위치에 있습니다.",
        "파이썬은 배우기 쉬운 프로그래밍 언어입니다."
    ]
    
    query = "바다에 사는 동물은?"
    
    print("1. OpenAI를 사용하여 벡터 임베딩 생성 중...")
    all_texts_to_embed = corpus + [query]
    vectors, total_tokens = await embed_texts_by_openai(all_texts_to_embed)
    
    corpus_vectors = vectors[:-1]
    query_vector = vectors[-1]
    
    print(f"총 {total_tokens} 토큰 사용됨.")

#     corpus_vectors = np.array([
#     # "상어는 바다의 포식자입니다."의 벡터 표현 (예시)
#     [0.015, -0.023, 0.088, 0.041],
#     # "고래는 매우 큰 포유류입니다."의 벡터 표현 (예시)
#     [0.031, -0.011, 0.075, 0.062],
#     # "기린은 목이 긴 동물로, 주로 아프리카 초원에 서식합니다."의 벡터 표현 (예시)
#     [-0.045, 0.005, -0.091, 0.019],
#     # "상어와 고래는 모두 바다에 사는 생물이지만..."의 벡터 표현 (예시)
#     # '상어', '고래', '바다' 키워드가 포함되어 첫 번째, 두 번째 벡터와 유사한 경향을 보일 수 있습니다.
#     [0.025, -0.018, 0.081, 0.055],
#     # "파이썬은 배우기 쉬운 프로그래밍 언어입니다."의 벡터 표현 (예시)
#     # 다른 문서들과 의미적으로 거리가 멀기 때문에 값의 패턴이 다를 수 있습니다.
#     [-0.078, 0.092, -0.053, -0.067]
# ])
#     query_vector = np.array([0.021, -0.015, 0.079, 0.051])

    print("\n2. BM25를 사용하여 임베딩 생성 중...")
    bm25_model = await embed_text_by_bm25(corpus)

    query_bm25_scores = bm25_model.get_scores(tokenize_before_bm25([query])[0])

    # 3. HybridScorer 초기화 (BM25: 0.3, Vector: 0.7)
    print("\n3. HybridScorer 초기화 및 인덱싱...")
    scorer = HybridScorer(texts=corpus, vector_embeddings=corpus_vectors, bm25_weight=0.3, vector_weight=0.7)

    # 4. 하이브리드 검색 수행
    print(f"\n4. 쿼리 \"{query}\"로 하이브리드 검색 수행...")
    results = scorer.search(query_bm25=query_bm25_scores, query_vector=query_vector)

    print("\n=== 검색 결과 (점수 높은 순) ===")
    for doc, score in results:
        print(f"점수: {score:.4f}\t문서: {doc}")

if __name__ == "__main__":
    asyncio.run(main())