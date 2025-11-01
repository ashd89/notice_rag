import base64
from typing import List, Dict, Any, Optional
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from apps.deps import get_settings


LayoutBlock = Dict[str, Any]

async def azure_read_layout(img_b64: str, lang: Optional[str] = None):
    """
    Azure Document Intelligence(구 Form Recognizer) v4.0
    prebuilt-layout 모델로 표/문단/라인/토큰까지 인식.
    반환: AnalyzeResult (SDK 객체)
    """
    cfg = get_settings()
    print("Azure Document Intelligence Endpoint:", cfg.azure_docint_endpoint)
    print("Azure Document Intelligence Key:", cfg.azure_docint_key is not None)
    client = DocumentIntelligenceClient(
        endpoint=cfg.azure_docint_endpoint,
        credential=AzureKeyCredential(cfg.azure_docint_key),
    )

    img_bytes = base64.b64decode(img_b64)

    # 모델: "prebuilt-layout" (표/레이아웃 인식에 최적)
    # content_type 은 바이너리면 "application/octet-stream"
    poller = await client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=img_bytes,
        content_type="application/octet-stream",
        locale=lang,
    )
    result = await poller.result()
    await client.close()
    return result

def normalize_from_azure(result) -> List[LayoutBlock]:
    """
    Azure AnalyzeResult -> 공통 블록(List[LayoutBlock])으로 정규화.
    - paragraphs: result.paragraphs
    - tables: result.tables (cells를 row/col 인덱스로 재구성)
    bbox는 normalized 좌표(0~1)로 변환.
    """
    blocks: List[LayoutBlock] = []

    # 페이지 크기 map (bbox 정규화용)
    page_size = {}
    for p in getattr(result, "pages", []) or []:
        # 일부 스키마: width/height/unit 존재
        w = float(getattr(p, "width", 0.0) or 0.0)
        h = float(getattr(p, "height", 0.0) or 0.0)
        page_size[p.page_number] = (w, h)

    def norm_bbox(bbox_points, page_number: int):
        # result.bbox가 [x,y] 좌표 리스트인 경우를 가정 (사변형)
        # (x0,y0,x1,y1)로 축약 + 0~1 정규화
        if not bbox_points:
            return [0, 0, 0, 0]
        xs = [bbox_points[i] for i in range(0, len(bbox_points), 2)]
        ys = [bbox_points[i] for i in range(1, len(bbox_points), 2)]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        w, h = page_size.get(page_number, (0.0, 0.0))
        if w > 0 and h > 0:
            return [x0 / w, y0 / h, x1 / w, y1 / h]
        return [x0, y0, x1, y1]

    # 문단(available in v4)
    for para in getattr(result, "paragraphs", []) or []:
        text = (para.content or "").strip()
        if not text:
            continue
        page_num = getattr(para, "spans", [None])[0].offset if getattr(para, "spans", None) else 1
        # 일부 버전에서는 paragraph.region이나 bounding_regions로 bbox 접근
        # bounding_regions: [{page_number, polygon}]
        if getattr(para, "bounding_regions", None):
            br = para.bounding_regions[0]
            bbox = norm_bbox(getattr(br, "polygon", None), int(br.page_number))
            page_idx = int(br.page_number) - 1
        else:
            bbox = [0, 0, 0, 0]
            page_idx = 0

        blocks.append({
            "type": "paragraph",
            "text": text,
            "bbox": bbox,
            "page": page_idx,
            "confidence": float(getattr(para, "confidence", 0.0) or 0.0),
            "extras": {},
        })

    # 테이블
    for table in getattr(result, "tables", []) or []:
        # 행/열 개수
        row_count = int(getattr(table, "row_count", 0) or 0)
        col_count = int(getattr(table, "column_count", 0) or 0)
        rows = [["" for _ in range(col_count)] for __ in range(row_count)]

        # 셀 채우기
        for cell in getattr(table, "cells", []) or []:
            r = int(getattr(cell, "row_index", 0) or 0)
            c = int(getattr(cell, "column_index", 0) or 0)
            txt = (getattr(cell, "content", "") or "").strip()
            # 병합 셀(rspan/cspan)은 extras에 기록해도 됨
            rows[r][c] = txt

        # bbox: bounding_regions 첫 번째 사용
        if getattr(table, "bounding_regions", None):
            br = table.bounding_regions[0]
            bbox = norm_bbox(getattr(br, "polygon", None), int(br.page_number))
            page_idx = int(br.page_number) - 1
        else:
            bbox = [0, 0, 0, 0]
            page_idx = 0

        table_text = "\n".join("\t".join(r) for r in rows)
        blocks.append({
            "type": "table",
            "text": table_text,
            "bbox": bbox,
            "page": page_idx,
            "confidence": float(getattr(table, "confidence", 0.0) or 0.0),
            "extras": {
                "rows": rows,
                "row_count": row_count,
                "column_count": col_count,
            },
        })

    return blocks