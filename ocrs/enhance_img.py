import asyncio
import logging
import aiohttp
import base64
from bs4 import BeautifulSoup
from typing import Optional
from .ocr_schema import LayoutBlock, flatten_blocks
from .azure_ocr import azure_read_layout, normalize_from_azure

from typing import Literal, List, Dict, Any, Optional, Tuple


__all__ = ["enhance_html_with_ocr"]

async def enhance_html_with_ocr(
    html: str,
    lang: str = 'ko',
    return_format: Literal["text", "json", "map"] = "text",
    ocr_timeout: int = 400,  # 개별 OCR 작업에 대한 타임아웃 (초)
    timeout_log_file: str = "timeout_images.log", # 타임아웃 로그 파일 경로
    failure_log_file: str = "ocr_failures.log" # 전체 실패 로그 파일 경로
    #todo 구분자
) -> str | dict:
    soup = BeautifulSoup(html, "html.parser")
    img_tags = soup.find_all("img")
    
    # 1. 고유한 이미지 소스(src)만 추출하여 중복 OCR 요청 방지
    unique_srcs = set()
    for img in img_tags:
        src = img.get('src', '')
        if src:  # 비어있지 않은 src만 추가
            unique_srcs.add(src)

    # 순서를 유지하고 인덱스로 접근하기 위해 리스트로 변환
    unique_srcs_list = list(unique_srcs)
    
    print(f"Found {len(img_tags)} image tags, processing {len(unique_srcs_list)} unique images for OCR.")

    # 1. 하나의 ClientSession을 생성하여 모든 HTTP 요청에 재사용합니다.
    async with aiohttp.ClientSession() as session:
        ocr_tasks = []
        for src in unique_srcs_list:
            if src.startswith('data:image'):
                # 기존 Base64 데이터 URI 처리
                try:
                    header, img_b64 = src.split(',', 1)
                    task = extract_layout_from_image(img_b64, src, lang=lang)
                    # wait_for를 사용하여 각 태스크에 타임아웃 적용
                    ocr_tasks.append(asyncio.wait_for(task, timeout=ocr_timeout))
                except ValueError:
                    logging.warning(f"Skipping invalid data URI: {src[:50]}...")
            elif src.startswith(('http://', 'https://')):
                # URL 처리 (재사용하는 session 객체를 전달)
                task = download_and_extract_layout(session, src, lang=lang)
                # wait_for를 사용하여 각 태스크에 타임아웃 적용
                ocr_tasks.append(asyncio.wait_for(task, timeout=ocr_timeout))


        # 병렬 처리
        if ocr_tasks:
            ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)
        else:
            ocr_results = []
            
    valid_ocr_results = []

    for i, result in enumerate(ocr_results): # i는 unique_srcs_list의 인덱스와 일치
        if isinstance(result, Exception):
            # 실패한 작업에 해당하는 이미지 src 가져오기
            failed_src = unique_srcs_list[i]

            # 타임아웃 예외를 별도로 처리하여 로그를 남김
            if isinstance(result, asyncio.TimeoutError):
                error_message = f"OCR task timed out after {ocr_timeout} seconds for src: {failed_src}"
                logging.error(error_message)
                # 타임아웃된 이미지 src를 파일에 기록
                with open(timeout_log_file, "a", encoding="utf-8") as f:
                    f.write(f"{failed_src}\n")
            else:
                error_message = f"OCR failed for src: {failed_src} with error: {result}"
                logging.error(error_message)
            
            # 모든 실패 사례(타임아웃 포함)를 별도 로그 파일에 상세히 기록
            with open(failure_log_file, "a", encoding="utf-8") as f:
                f.write(f"SRC: {failed_src}\nERROR: {result}\n---\n")
        elif isinstance(result, dict) and result.get("blocks"):
            valid_ocr_results.append(result)
        else:
            logging.warning(f"OCR returned empty or invalid result for image {i+1}")

    if return_format == "json":
       logging.info(f"OCR JSON {len(valid_ocr_results)} images successfully.")
       return {
           # OCR을 수행한 후, img 태그를 제외한 텍스트를 추출합니다.
           "original_text": (lambda s: (
               [img.decompose() for img in s.find_all('img')], 
               s.get_text(" ", strip=True)
           )[1])(BeautifulSoup(html, 'html.parser')),
           "original_text": original_text,
           "ocr_texts": valid_ocr_results
       }

    if return_format == "map":
        ocr_map = {}
        for item in valid_ocr_results:
            if not item["blocks"]:
                continue
            flat = flatten_blocks(item["blocks"])
            if flat.strip():
                ocr_map[item['src']] = flat
        return ocr_map

    # 텍스트 모드 블록을 평탄화
    # OCR을 수행한 후, img 태그를 제외한 텍스트를 추출합니다.
    original_text = (lambda s: (
        [img.decompose() for img in s.find_all('img')],
        s.get_text(" ", strip=True)
    )[1])(BeautifulSoup(html, 'html.parser'))
    parts = [original_text] if original_text.strip() else []
    for item in valid_ocr_results:
        if not item["blocks"]:
            continue
        flat = flatten_blocks(item["blocks"])
        if flat.strip():
            parts.append(f"image: {item['src']} - text:\n{flat}")
    return "\n\n".join(parts)

    
async def extract_layout_from_image(
    img_b64: str,
    original_src: str,
    lang: Optional[str] = None
) -> List[LayoutBlock]:
    print("Extracting layout from image with Azure OCR...")
    resp = await azure_read_layout(img_b64, lang=lang)
    blocks = normalize_from_azure(resp)
    return {
        "src": original_src,
        "blocks": blocks
    }

async def download_and_extract_layout(
    session: aiohttp.ClientSession,
    img_url: str,
    lang: Optional[str] = None
) -> List[LayoutBlock]:
    """URL에서 이미지를 다운로드하고 Base64로 인코딩한 후 OCR을 수행합니다."""
    print(f"Downloading image from URL: {img_url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    # 전달받은 세션을 사용하여 연결합니다.
    async with session.get(img_url, headers=headers) as response:
        response.raise_for_status()  # HTTP 오류가 있으면 예외 발생
        img_bytes = await response.read() # aiohttp에서 받은 resonse는 bytes 형태
        img_b64 = base64.b64encode(img_bytes).decode('utf-8') # bytes -> base64 encoding
        return await extract_layout_from_image(img_b64, original_src=img_url, lang=lang)