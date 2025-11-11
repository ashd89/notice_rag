import sys
import os
import asyncio

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from embeds.embed import embed_text

async def main():
    print("=== imbedding 테스트 ===")
    test_text = []
    test_text.append("상어가 고래를 잡아 먹는다")
    test_text.append("고래는 상어가 무섭다")
    test_text.append("기린은 고래의 친구다")

    vectors, total_tokens = await embed_text(test_text)
    for i, vec in enumerate(vectors):
        print(f"\nText {i} Vector (first 5 dims): {vec[:5]}... (total dims: {len(vec)})")
    print(f"\nTotal tokens used for embedding: {total_tokens}")

if __name__ == "__main__":
    asyncio.run(main())