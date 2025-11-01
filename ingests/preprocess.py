# ingests/preprocess.py
import re
import unicodedata
from typing import List, Dict, Tuple, Optional, Callable, Any
from bs4 import BeautifulSoup, Tag


__all__ = [
    "normalize",
    "dedupe",
    "normalize_and_zip_table",
    "fix_numbers_units",
    "absorb_fragments",
    "group_by_headings",
    "default_token_len_estimator",
    "pack_by_tokens",
    "hard_split_by_tokens",
    "find_nearest_prev_chunk_index",
    "extract_text",
]

# =============== 공통 유틸 ===============
def extract_text(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    texts = []
    for tag in soup.find_all(["h1", "h2", "p", "li", "span"]):
        txt = tag.get_text(strip=True)
        if txt and txt not in texts:
          texts.append(txt)
    return texts


def normalize(s: str) -> str:
    """NFKC 정규화 + 줄바꿈 하이픈 제거 + 공백 정리."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"-\s*\n\s*", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def dedupe(items: List[str]) -> List[str]:
    """순서 보존 중복 제거."""
    seen = set()
    out = []
    for x in items:
        x2 = x.strip()
        if not x2:
            continue
        if x2 not in seen:
            seen.add(x2)
            out.append(x2)
    return out

# =============== 1) 테이블 라벨-값 합치기 ===============
def normalize_and_zip_table(lines: List[str]) -> List[str]:
    """
    아주 일반적인 헤더-행 패턴을 감지해 '라벨: 값 | 라벨: 값'으로 합칩니다.
    - 헤더 후보는 한 줄에 10자 이하, 특수기호 적고 명사 위주인 경우로 휴리스틱.
    - 값 행은 헤더 수와 비슷한 개수의 짧지 않은 문장들을 찾습니다.
    - 실패하면 원본을 그대로 반환(보수적).
    """
    out: List[str] = []
    i = 0
    n = len(lines)

    def is_header_like(s: str) -> bool:
        s2 = s.strip()
        if len(s2) == 0 or len(s2) > 10:
            return False
        # 너무 기호만 있는 경우 제외
        if re.fullmatch(r"[\W_]+", s2):
            return False
        # 흔한 헤더 단어들에 가산점(한국어 일반)
        hint = ["구분", "항목", "제목", "분야", "내용", "상금", "수", "인원", "대상"]
        score = 0
        for h in hint:
            if h in s2:
                score += 1
        # 짧고 깔끔한 문자열 허용
        return score > 0 or (len(s2) <= 4)

    while i < n:
        # 헤더 후보 2~5개 연속을 탐지
        j = i
        headers: List[str] = []
        while j < n and is_header_like(lines[j]):
            headers.append(lines[j].strip())
            j += 1
            if len(headers) >= 5:
                break

        # 헤더를 최소 2개 이상 찾았을 때만 시도
        if len(headers) >= 2:
            # 이어지는 값 행들을 모아본다: 값 라인도 너무 짧지 않도록
            rows: List[List[str]] = []
            k = j
            # 값 라인이 어느 정도 나열될 때까지(빈 줄/헤딩/단절 조건까지)
            while k < n:
                val_line = lines[k].strip()
                if not val_line:
                    break
                # 다음 헤딩/섹션 번호 등장하면 테이블 종료
                if re.match(r"^\d+\.\s*", val_line):
                    break
                # 단독 기호 라인, 너무 짧은 파편은 행 구분자로 취급
                if len(val_line) <= 1 and re.fullmatch(r"[*/()▶·\-]+", val_line):
                    k += 1
                    continue
                # 값 라인을 여러 개 모아 하나의 행으로 삼기(헤더 수에 맞춰 분배)
                # 휴리스틱: 쉼표/파이프/슬래시/스페이스로 split해서 덩어리 수가 헤더수와 비슷하면 1행으로 간주
                chunks = re.split(r"\s*[|,/]\s*|\s{2,}", val_line)
                chunks = [c for c in chunks if c]
                if len(chunks) >= len(headers):
                    rows.append(chunks[:len(headers)])
                else:
                    # 헤더 수에 못 미치면 다음 라인까지 합쳐서 채우기
                    lookahead = k + 1
                    acc = chunks[:]
                    while lookahead < n and len(acc) < len(headers):
                        nxt = re.split(r"\s*[|,/]\s*|\s{2,}", lines[lookahead].strip())
                        nxt = [c for c in nxt if c]
                        if not nxt:
                            break
                        acc += nxt
                        lookahead += 1
                    if len(acc) >= len(headers):
                        rows.append(acc[:len(headers)])
                        k = lookahead - 1  # lookahead가 한 칸 더 가 있으니 보정
                    else:
                        # 테이블로 보기 어려움 → 원본으로 내보내고 종료
                        rows = []
                        break
                k += 1

            if rows:
                # 헤더-값 라벨링으로 직렬화
                for r in rows:
                    labeled = [f"{headers[idx]}: {r[idx]}" for idx in range(len(headers))]
                    out.append(" | ".join(labeled))
                i = k
                continue  # 다음 구간으로
            # rows가 비면 테이블 아님 → 원본 라인 출력
        # 테이블 아님
        out.append(lines[i])
        i += 1

    return out

# =============== 2) 숫자/단위 결합 + 날짜 표준화 ===============
def fix_numbers_units(s: str) -> str:
    """
    - '100 만원' -> '100만원', '70 만원' 등 결합
    - '1 팀' -> '1팀', '2 ~ 5 명' -> '2~5명'
    - '’25. 9. 1.' -> '2025-09-01' 형식으로 표준화(’25 → 2025 추정)
    """
    t = normalize(s)
    # 숫자 + 공백 + (만원|원|팀|명|개|월|일)
    t = re.sub(r"(\d+)\s*(만원|원|팀|명|개|월|일|학점)", r"\1\2", t)
    # 범위 공백 정리: 2 ~ 5 -> 2~5
    t = re.sub(r"(\d+)\s*~\s*(\d+)", r"\1~\2", t)

    # 날짜 패턴들 정규화
    # ʼ25. 9. 1.  → 2025-09-01
    def _year_expand(m):
        yy = m.group(1)
        mm = int(m.group(2))
        dd = int(m.group(3))
        # '25 → 2025 가정 (20xx)
        year = 2000 + int(yy)
        return f"{year:04d}-{mm:02d}-{dd:02d}"

    t = re.sub(r"[’'](\d{2})\.\s*(\d{1,2})\.\s*(\d{1,2})\.", _year_expand, t)
    # 2025. 9. 1. → 2025-09-01
    def _ymd(m):
        y = int(m.group(1)); mm = int(m.group(2)); dd = int(m.group(3))
        return f"{y:04d}-{mm:02d}-{dd:02d}"
    t = re.sub(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.", _ymd, t)
    return t

# =============== 3) 파편 흡수 ===============
def absorb_fragments(lines: List[str], min_len: int = 5) -> List[str]:
    """
    아주 짧은 토큰/기호 라인을 이웃 문장에 합칩니다.
    - 앞줄이 있으면 앞줄 뒤에 붙이고, 없으면 다음 줄 앞에 붙임.
    """
    out: List[str] = []
    buf = ""
    frags = set(["*", "▶", "/", "(", ")", "-", ","])
    for i, s in enumerate(lines):
        ss = s.strip()
        if not ss:
            continue
        if len(ss) < min_len or ss in frags or re.fullmatch(r"[()/,.\-]+", ss):
            # 앞에 붙일 대상이 있으면 붙이기
            if out:
                out[-1] = (out[-1] + " " + ss).strip()
            else:
                buf = (buf + " " + ss).strip()
            continue
        # 누적 버퍼가 있으면 먼저 출력
        if buf:
            out.append(buf)
            buf = ""
        out.append(ss)
    if buf:
        if out:
            out[-1] = (out[-1] + " " + buf).strip()
        else:
            out.append(buf)
    return out

# =============== 4) 헤딩/섹션 분할 ===============
SUB_PREFIXES = (
    r"^\d+\)",          # 1), 2), ...
    r"^▶", r"^•",
    r"^[-–—]\s*",       # -, –, — (공백 유무 허용)
)

def group_by_headings(lines, headings_prefixes=tuple(str(i)+"." for i in range(1, 20))):
    sections = [[]]
    for s in lines:
        st = s.strip()
        if any(st.startswith(p) for p in headings_prefixes) \
           or re.match("|".join(SUB_PREFIXES), st):
            sections.append([s])
        else:
            sections[-1].append(s)
    return [sec for sec in sections if any(x.strip() for x in sec)]
# =============== 5) 토큰 길이 기준 재패킹 ===============
def default_token_len_estimator(text: str) -> int:
    """간단 추정: 글자수/4 ≈ 토큰 수."""
    return max(1, len(text) // 4)

def pack_by_tokens(
    sentences: List[str],
    max_tokens: int = 512,
    overlap_sents: int = 1,
    token_len: Callable[[str], int] = default_token_len_estimator
) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    def buf_len(sents: List[str]) -> int:
        return sum(token_len(x) for x in sents)

    for sent in sentences:
        if not sent.strip():
            continue
        candidate = buf + [sent]
        if buf_len(candidate) > max_tokens:
            if buf:
                chunks.append(" ".join(buf).strip())
                buf = buf[-overlap_sents:] if overlap_sents > 0 else []
                # 한 문장 자체가 너무 길면 강제 쪼개기(문자 기준)
                if token_len(sent) > max_tokens:
                    pieces = hard_split_by_tokens(sent, max_tokens, token_len)
                    for p in pieces[:-1]:
                        chunks.append(p)
                    buf = [pieces[-1]]
                else:
                    buf.append(sent)
            else:
                # 버퍼가 비어있는데도 초과 → 강제 분할
                pieces = hard_split_by_tokens(sent, max_tokens, token_len)
                chunks.extend(pieces[:-1])
                buf = [pieces[-1]]
        else:
            buf = candidate
    if buf:
        chunks.append(" ".join(buf).strip())
    return chunks

def hard_split_by_tokens(text: str, max_tokens: int, token_len: Callable[[str], int]) -> List[str]:
    """
    토크나이저가 없을 때의 안전 분할: 문자 길이를 이용해 대략 쪼갬.
    """
    approx_toks = token_len(text)
    if approx_toks <= max_tokens:
        return [text]
    # 문자 비율로 쪼개기
    n_parts = approx_toks // max_tokens + (1 if approx_toks % max_tokens else 0)
    step = max(1, len(text) // n_parts)
    parts = [text[i:i+step] for i in range(0, len(text), step)]
    return [p.strip() for p in parts if p.strip()]

# =============== 6) OCR을 가장 가까운 섹션 청크에 붙이기 ===============
def find_nearest_prev_chunk_index(img_src: str, chunks: List[str], html: str) -> int:
    """
    img_src가 등장하는 <img> 바로 '이전'의 텍스트를 DOM에서 찾아
    그 텍스트를 포함한 청크 인덱스를 추정. 실패시 마지막 청크.
    """
    soup = BeautifulSoup(html, "html.parser")
    target = None
    for img in soup.find_all("img"):
        src = (img.get("src") or "").strip()
        if src == img_src:
            target = img
            break
    if target is None:
        return max(0, len(chunks)-1)

    # 이전 텍스트 노드/태그를 역방향으로 탐색
    prev_text = None
    cur: Optional[Tag] = target
    while cur is not None:
        sib = cur.previous_sibling
        while sib is not None:
            if isinstance(sib, Tag):
                txt = normalize(sib.get_text(" ", strip=True))
                if txt:
                    prev_text = txt
                # sib 내부의 마지막 유의미 텍스트를 더 우선
                inner = [normalize(t.get_text(" ", strip=True)) for t in sib.find_all(True)]
                inner = [x for x in inner if x]
                if inner:
                    prev_text = inner[-1]
                if prev_text:
                    break
            sib = sib.previous_sibling
        if prev_text:
            break
        cur = cur.parent if isinstance(cur, Tag) else None

    if not prev_text:
        return max(0, len(chunks)-1)

    # prev_text를 포함하는(부분 일치) 첫 청크를 찾는다. 없으면 유사도 기반으로 대강 매칭.
    for idx, ch in enumerate(chunks):
        if prev_text in ch:
            return idx

    # 부분 일치가 안되면 공백 제거 후 서브스트링 길이 비교로 가장 비슷한 후보를 찾기
    pt = prev_text.replace(" ", "")
    best_i, best_overlap = len(chunks)-1, 0
    for idx, ch in enumerate(chunks):
        ct = ch.replace(" ", "")
        # 공통 부분 길이를 근사(아주 단순)
        overlap = _longest_common_subseq_len(pt, ct)
        if overlap > best_overlap:
            best_overlap = overlap
            best_i = idx
    return best_i

def _longest_common_subseq_len(a: str, b: str) -> int:
    # O(n*m) 간단 LCS 길이(짧은 텍스트에만 사용하므로 충분)
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]