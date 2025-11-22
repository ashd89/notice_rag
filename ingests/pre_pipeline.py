# pre_pipeline module
from .preprocess import * 
from typing import List, Dict, Tuple, Optional, Callable, Any

__all__ = [
    # 파이프라인 엔드포인트
    "build_chunks_from_blocks_and_ocr",
]

def build_chunks_from_blocks_and_ocr(
    html: str,
    extracted_texts: List[str],
    ocr_map: Dict[str, str],
    max_tokens: int = 256,
    overlap_sents: int = 1,
    token_len: Callable[[str], int] = default_token_len_estimator
) -> List[str]:
    """
    0) 입력: 이미 추출한 문장/문단 리스트(extracted_texts), html, ocr_map(src->text)
    1) 테이블 라벨-값 합치기
    2) 숫자/단위 결합 + 날짜 표준화
    3) 파편 흡수
    4) 헤딩 기준 섹션화
    5) 섹션별 토큰 길이 재패킹
    6) OCR을 가장 가까운 이전 섹션 청크에 붙이기
    7) 중복 제거
    """
    # 1
    blocks = normalize_and_zip_table([normalize(x) for x in extracted_texts])
    # 2
    blocks = [fix_numbers_units(b) for b in blocks]
    # 3
    blocks = absorb_fragments(blocks, min_len=5)
    # 4
    sections = group_by_headings(blocks)
    # 5
    chunks: List[str] = []
    for sec in sections:
        chunks.extend(pack_by_tokens(sec, max_tokens=max_tokens, overlap_sents=overlap_sents, token_len=token_len))
    # 6
    for img_src, ocr_text in (ocr_map or {}).items():
        if not ocr_text:
            continue
        
        # OCR 텍스트도 HTML 본문과 동일하게 1~5단계 전처리 수행
        # 1. OCR 텍스트를 라인으로 분리하고 테이블 처리
        ocr_blocks = normalize_and_zip_table([normalize(x) for x in ocr_text.split('\n')])
        # 2. 숫자/단위 정규화
        ocr_blocks = [fix_numbers_units(b) for b in ocr_blocks]
        # 3. 짧은 파편 흡수
        ocr_blocks = absorb_fragments(ocr_blocks, min_len=5)
        # 4. 헤딩 기준 섹션화
        ocr_sections = group_by_headings(ocr_blocks)
        # 5. 섹션별 토큰 길이 재패킹
        ocr_chunks: List[str] = []
        for sec in ocr_sections:
            ocr_chunks.extend(pack_by_tokens(sec, max_tokens=max_tokens, overlap_sents=0, token_len=token_len))
        
        if not ocr_chunks:
            continue

        # 분할된 OCR 청크들을 기존 청크에 병합/추가
        first_ocr_chunk = "\n\n[IMAGE_TEXT]\n" + ocr_chunks[0]
        
        if not chunks:
            chunks.append(first_ocr_chunk.strip())
        else:
            idx = find_nearest_prev_chunk_index(img_src, chunks, html)
            if first_ocr_chunk not in chunks[idx]:
                chunks[idx] += first_ocr_chunk
        
        # 나머지 OCR 청크들은 새로운 청크로 추가
        for ocr_part in ocr_chunks[1:]:
            chunks.append(("\n\n[IMAGE_TEXT]\n" + ocr_part).strip())
    # 7
    chunks = dedupe([c.strip() for c in chunks if c.strip()])
    return chunks
