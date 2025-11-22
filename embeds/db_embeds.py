import sys
import os
import asyncio
import asyncpg
from pgvector.asyncpg import register_vector

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# from embeds.embed import embed_texts_by_openai
from apps.deps import get_settings
# 전처리 및 OCR 관련 모듈 import
from ingests.preprocess import extract_text
from ingests.pre_pipeline import build_chunks_from_blocks_and_ocr
from ocrs.enhance_img import enhance_html_with_ocr
from embeds.embed import embed_texts_by_openai

cfg = get_settings()

async def get_db_connection():
    return await asyncpg.connect(
        user=cfg.db_user,
        password=cfg.db_password,
        database=cfg.db_name,
        host=cfg.db_host,
        port=cfg.db_port
    )

async def main():
    print("=== 공지사항 임베딩 생성 및 DB 저장 시작 ===")
    
    conn = None
    try:
        conn = await get_db_connection()
        print("1. 데이터베이스에서 임베딩이 필요한 공지사항을 가져옵니다.")

        await register_vector(conn) # 임베딩 생성 및 저장 함수 호출
        
        # embedded가 TRUE가 아닌 공지사항만 선택
        records = await conn.fetch("SELECT id, title, html FROM announcement WHERE embedded IS NOT TRUE ORDER BY id ASC")
        
        if not records:
            print("새롭게 임베딩할 공지사항이 없습니다.")
            return
        
        print(f"총 {len(records)}개의 공지사항에 대한 임베딩을 생성합니다.")

        # 임베딩할 모든 청크 텍스트와 메타데이터를 수집할 리스트
        all_chunks_to_embed = []
        # DB에 저장할 데이터를 준비하는 리스트
        db_insert_data = []
        
        # 각 공지사항을 순회하며 청크 생성
        for i, record in enumerate(records):
            print(f"  - [{i+1}/{len(records)}] 공지사항 ID {record['id']} 처리 중...")
            html_content = record['html'] or ""
            title = record['title'] or ""

            # --- 디버깅용 코드 추가 ---
            if record['id'] == 4:
                print(f"    [디버그] ID 4의 HTML 내용 길이: {len(html_content)}")
            # -----------------------
            
            # 1. HTML에서 이미지 OCR 수행
            ocr_map = await enhance_html_with_ocr(html_content, return_format="map")
            
            # 2. HTML에서 텍스트 추출
            extracted_texts = extract_text(html_content)
            
            # 3. OCR 결과와 텍스트를 합쳐 청크 생성
            chunks = build_chunks_from_blocks_and_ocr(
                html=html_content,
                extracted_texts=extracted_texts,
                ocr_map=ocr_map
            )

            print(f"    (생성된 청크 수: {len(chunks)})")
            for chunk_idx, chunk_content in enumerate(chunks):
                # 문맥 보강을 위해 제목을 각 청크에 메타데이터로 추가
                added_chunk_text = f"{title} \n {chunk_content}"
                all_chunks_to_embed.append(added_chunk_text)
                print(f"      [청크 {chunk_idx}] (길이: {len(added_chunk_text)}) : {added_chunk_text.replace(chr(10), ' ')}")
                
                # DB 저장용 데이터 준비 (벡터는 나중에 채움)
                db_insert_data.append({
                    "announcement_id": record['id'],
                    "length": len(added_chunk_text),
                    "chuncked_text" : added_chunk_text,
                    "embedding": None
                })
            
        if not all_chunks_to_embed:
            print("임베딩할 청크가 없습니다.")
            return

        print("\n2. OpenAI를 사용하여 모든 청크의 벡터 임베딩 생성 중...")
        vectors, total_tokens = await embed_texts_by_openai(all_chunks_to_embed)
        print(f"총 {len(vectors)}개의 벡터 생성, {total_tokens} 토큰 사용됨.")


        
        # 생성된 벡터를 db_insert_data에 할당
        for i, vector in enumerate(vectors):
            db_insert_data[i]['embedding'] = vector
            
        print("\n3. 생성된 임베딩을 데이터베이스에 저장합니다.")
        # executemany를 사용하여 여러 레코드를 한 번에 삽입
        insert_query = """
            INSERT INTO announcement_embedding (announcement_id, length, chuncked_text, embedding)
            VALUES ($1, $2, $3, $4)
        """
        
        # executemany에 맞는 튜플 리스트로 변환
        excute_tuple = [(d['announcement_id'], d['length'], d['chuncked_text'], d['embedding']) for d in db_insert_data]
        
        await conn.executemany(insert_query, excute_tuple)
        print(f"총 {len(excute_tuple)}개의 청크 임베딩을 데이터베이스에 성공적으로 저장했습니다.")

        # 처리된 announcement_id 목록을 가져와 중복을 제거
        processed_announcement_ids = list(set(d['announcement_id'] for d in db_insert_data))

        if processed_announcement_ids:
            print("\n4. 원본 공지사항의 'embedded' 상태를 TRUE로 업데이트합니다.")
            update_query = "UPDATE announcement SET embedded = TRUE WHERE id = ANY($1)"
            await conn.execute(update_query, processed_announcement_ids)
            print(f"총 {len(processed_announcement_ids)}개 공지사항의 상태를 업데이트했습니다.")
            
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        if conn:
            await conn.close()
            print("데이터베이스 연결을 종료했습니다.")

if __name__ == "__main__":
    asyncio.run(main())