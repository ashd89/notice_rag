# pre_pipeline module
from preprocess import * 
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
        idx = find_nearest_prev_chunk_index(img_src, chunks, html)
        ocr_blob = "\n\n[IMAGE_TEXT]\n" + normalize(ocr_text)
        chunks[idx] = (chunks[idx] + ocr_blob) if ocr_blob not in chunks[idx] else chunks[idx]
    # 7
    chunks = dedupe([c.strip() for c in chunks if c.strip()])
    return chunks


