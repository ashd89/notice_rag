from typing import List, Tuple
import re
from konlpy.tag import Okt
from rank_bm25 import BM25Okapi
from apps.deps import get_openai_client, get_settings


BATCH_SIZE = 256  # 배치 크기 조절
okt = Okt()

async def embed_texts_by_openai(
    texts: List[str],
) -> Tuple[List[List[float]], int]:
    client = get_openai_client()
    model = get_settings().embed_model
    vectors: List[List[float]] = []
    total_tokens = 0

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        resp = await client.embeddings.create(model=model, input=batch)
        vectors.extend([d.embedding for d in resp.data])
        # resp.usage = 사용된 토큰 수 
        if resp.usage:
            total_tokens += resp.usage.total_tokens or 0

    return vectors, total_tokens

async def embed_text_by_bm25(
        texts: List[str],
) -> BM25Okapi:
    tokenized_texts = tokenize_before_bm25(texts)
    bm25 = BM25Okapi(tokenized_texts)
    return bm25

def tokenize_before_bm25(texts: List[str]) -> List[List[str]]:
    tokenized_docs: List[List[str]] = []
    korean_pattern = re.compile(r"[ㄱ-ㅎㅏ-ㅣ가-힣]+")

    for doc in texts:
        if korean_pattern.search(doc):
            # 한글 텍스트: Okt 형태소 분석기 사용
            tokenized_docs.append(okt.morphs(doc))
        else:
            # 영문 텍스트 (또는 한글이 없는 경우): 소문자 변환 후 공백 기준 분리
            tokenized_docs.append(doc.lower().split())
    return tokenized_docs