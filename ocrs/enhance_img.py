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
    #todo 구분자
) -> str | dict:
    soup = BeautifulSoup(html, 'html.parser')
    for img in soup.find_all('img'):
        # img 태크 항목 삭제
        img.decompose()
    original_text = soup.get_text(" ", strip=True)
    print("Original Text Extracted from HTML:", original_text)

    soup = BeautifulSoup(html, "html.parser")
    img_tags = soup.find_all("img")
    print(f"Found {len(img_tags)} image tags for OCR processing.")

    ocr_tasks = []
    for img in img_tags:
        src = img.get('src', '')
        print("Processing image src:", src)
        if src.startswith('data:image'):
            # 기존 Base64 데이터 URI 처리
            try:
                header, img_b64 = src.split(',', 1)
                ocr_tasks.append(extract_layout_from_image(img_b64, src, lang=lang))
            except ValueError:
                logging.warning(f"Skipping invalid data URI: {src[:50]}...")
        elif src.startswith(('http://', 'https://')):
            # URL 처리
            ocr_tasks.append(download_and_extract_layout(src, lang=lang))


    # 병렬 처리
    ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)
    valid_ocr_results = []

    for i, result in enumerate(ocr_results):
        if isinstance(result, Exception):
            logging.error(f"OCR failed for image {i+1}: {result}")
        elif isinstance(result, dict) and result.get("blocks"):
            valid_ocr_results.append(result)
        else:
            logging.warning(f"OCR returned empty or invalid result for image {i+1}")

    if return_format == "json":
       logging.info(f"OCR JSON {len(valid_ocr_results)} images successfully.")
       return {
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
    parts = [original_text] if original_text else []
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
    img_url: str,
    lang: Optional[str] = None
) -> List[LayoutBlock]:
    """URL에서 이미지를 다운로드하고 Base64로 인코딩한 후 OCR을 수행합니다."""
    print(f"Downloading image from URL: {img_url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    async with aiohttp.ClientSession() as session:
        # User-Agent 헤더를 추가하여 서버 차단을 우회합니다.
        async with session.get(img_url, headers=headers) as response:
            response.raise_for_status()  # HTTP 오류가 있으면 예외 발생
            img_bytes = await response.read() # aiohttp에서 받은 resonse는 bytes 형태
            img_b64 = base64.b64encode(img_bytes).decode('utf-8') # bytes -> base64 encoding
            return await extract_layout_from_image(img_b64, original_src=img_url, lang=lang)