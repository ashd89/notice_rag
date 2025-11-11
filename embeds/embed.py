from typing import List, Tuple
from apps.deps import get_openai_client, get_settings    

BATCH_SIZE = 256  # 배치 크기 조절


async def embed_text(
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