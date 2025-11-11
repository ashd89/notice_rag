# run_preprocess.py

import sys
import os

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import asyncio
from preprocess import extract_text
from pre_pipeline import build_chunks_from_blocks_and_ocr
from ocrs.enhance_img import enhance_html_with_ocr
async def main():
    html = """<p><span><b>&lt;지원자가 많아 조기마감 되었습니다. 감사합니다.&gt; </b></span></p><p><span><br /></span></p><p><span>안녕하세요. 인공지능혁신융합대학사업단 입니다. </span></p><p><span><br /></span></p><p><span>2025학년도 1학기 인공지능 분야 운영 지원 교과목 수업조교를 아래와 같이 모집하오니 </span></p><p><span>관심있는 학생은 기한 내 신청하시기 바랍니다. </span></p><p><br /></p><p><span>1.  </span><span>운영기간 </span><span>: 25. 3. 1. ~ 6. 23.(1학기, 4개월) </span></p><p><span><br /></span></p><p><span><b><span>2. 선발인원: 과목별 1명(총 2명) </span></b></span></p><p><span><span>ㅇ 인공지능수학기초 1분반 </span></span><span>(화 2,3,4) </span><span>(최창열) </span><span>  </span></p><p><span><span>ㅇ 인공지능수학기초 2분반(수 2,3,4) </span></span><span>(최창열) </span></p><p><br /></p><p><span>3.  </span><span>선발기준  </span><span>  </span></p><p><span><b><span>  </span></b>-  </span><span>현재  </span><span>재학중인 <u><span>학부생 또는 대학원 재학생 </span></u></span><span><span>( </span><span>* </span></span><span>졸업생 </span><span>,  </span><span>휴학생 불가 </span><span>) </span></p><p><span>  </span><span>-  </span><span>학부생인 경우 <span><span>고학년, 개설학부생이 아니어도 가능 </span></span></span></p><p><span> -  </span><span>대학원생은 석/ </span><span>박사 구분 없으며,  </span><span>연구과제에 참여해서 급여를 받는 경우도 지원 가능 </span></p><p><span> -  </span><span>교내 대학원 장학금 수혜자의 경우 사업단에서 중복수혜 여부 검토 후 선발 예정 </span></p><p><span>   </span><span>※ </span><span>수업조교로 선발된 학생은 아래 인공지능사업단에서 운영하는 프로그램에 중복 참여할 수 없음 </span></p><p><span> ( </span><span>학생 참여 재직자교육 </span><span>,  </span><span>전공심화멘토링 </span><span>, AICOSS 서포터즈 등 사업단에서 근로 장학금을 받는 프로그램 </span><span>) </span></p><p><br /></p><p><span>4.  </span><span>주요업무 </span><span>: 운영 지원 교과목 강의보조 및 사업단 관련 업무 보조 </span></p><p><span> -  </span><span>사업단 관련 업무 </span><span>: 종강 후 교과목 만족도 조사 안내 </span></p><p><span> - 필수 근로시간: 매 월 <b><span>20시간  </span></b></span></p><p><span><span> - 수업조교를 희망하는 학생은 담당 교수님과 활동 기간(요일) 및 시간, 담당 업무를 사전 협의함 </span></span></p><p><span> - 수업조교 활동을 불성실하게 이행하는 경우 수업조교 선발을 취소할 수 있음 </span></p><p><span><br /></span></p><p><span><b>5. 장학금: 매 월 500,000원/1인(1개월 기준) </b></span></p><p><span><span> - 총액: 500,000/1인*4개월=2,000,000원(1학기 활동 기준) </span></span></p><p><span><span> - 장학금은 매 월 말 활동보고서 취합 후 다음 달에 지급됨(예: 3월 활동 시 4월 지급) </span></span></p><p><span><br /></span></p><p><span><b>6. 지원 서류 제출 안내 </b></span></p><p><span><b> - 제출서류: 수업조교 신청 양식(엑셀)   </b></span></p><p><span><span>   ※ 메일 제목: 수업조교 신청(이름_과목명) </span></span></p><p><span>   ※ <span><span>활동가능시간에 요일 및 시간 필수 작성 </span></span></span></p><p><span> -  </span><span>제출기간 </span><span>: 2025. 2. 19.(수 </span><span>)  </span><span>오전 10시까지 </span></p><p><span><b> - 제출처: 담당자 이메일(ljy@uos.ac.kr)  </b></span></p><p><b><br /></b></p><p><b><span>★ 신청 마감 후 교수님께서 대상자 개별 연락 예정 </span></b></p><p><span><b>★ 지원자가 많을 경우 조기 마감될 수 있으며, 적격자가 없을 시 선발하지 않을 수 있음 </b></span><span><b>( </b></span><b><span>조기 마감 시 공지 예정) </span></b></p><p><br /></p><p><span><b>※ 자세한 내용 보기:  https://www.sorisaem.net/1010/1187 </b></span></p><p><br /></p><p><br /></p><p><img src="http://volunteer.uos.ac.kr/webedit_naver/upload/250228/20250228102044_7cd69ed98e0cc55b8074.png" title="20250228102044_7cd69ed98e0cc55b8074.png" /><br />  </p>"""
    # 1. HTML에서 이미지 OCR 수행하여 ocr_map 생성
    ocr_map = await enhance_html_with_ocr(html, return_format="map")
    # print("--- OCR 결과 ---")
    # print(ocr_map)
    # print("--------------------")

    extracted_texts = extract_text(html)

    chunks = build_chunks_from_blocks_and_ocr(
        html=html,
        extracted_texts=extracted_texts,
        ocr_map=ocr_map,
        max_tokens=512,
        overlap_sents=1,
    )

    print(f"\n청크 수: {len(chunks)}")
    for i, ch in enumerate(chunks):
        print(f"\n[{i}] (len={len(ch)})")
        print(ch)

if __name__ == "__main__":
    asyncio.run(main())
