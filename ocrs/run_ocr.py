import sys
import os
import asyncio

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from enhance_img import enhance_html_with_ocr

async def main():
    print("=== OCR 테스트 ===")
    ocr_url = """<p><img src="http://volunteer.uos.ac.kr/webedit_naver/upload/250225/20250225132037_d94e4a9c369c160af692.jpg" title="20250225132037_d94e4a9c369c160af692.jpg" /><br />  </p><p><img src="http://volunteer.uos.ac.kr/webedit_naver/upload/250225/20250225132044_639897a87d8b66469f41.jpg" title="20250225132044_639897a87d8b66469f41.jpg" /><br />  </p>"""

    text: str = await enhance_html_with_ocr(ocr_url, return_format="text")
    print(f"\nOCR Text:\n{text}")

if __name__ == "__main__":
    asyncio.run(main())
    