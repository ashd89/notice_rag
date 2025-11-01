from typing import Literal, List, Dict, Any, Optional, Tuple

BlockType = Literal["paragraph", "table", "table_cell", "title", "list", "figure", "footnote", "equation", "header", "footer"]

class LayoutBlock(Dict[str, Any]):
    """
    {
      "type": BlockType,
      "text": str,               # 테이블 셀/수식도 text를 채워두고, 추가 필드는 extras에
      "bbox": [x0,y0,x1,y1],     # 선택: 픽셀 좌표
      "page": int,               # 선택: 페이지/이미지 인덱스
      "confidence": float,       # 선택
      "extras": dict             # 표 구조, 수식 LaTeX, merged cell info 등
    }
    """

def flatten_blocks(blocks: List[LayoutBlock]) -> str:
    """문단/표/수식 순서로 텍스트만 평탄화(빠른 하위호환)."""
    parts = []
    for b in blocks:
        t = b.get("text","").strip()
        if not t: 
            continue
        if b["type"] == "table":
            parts.append("[TABLE]\n" + t + "\n[/TABLE]")
        elif b["type"] in ("equation",):
            parts.append("[$] " + t + " [/$]")
        else:
            parts.append(t)
    return "\n".join(parts)