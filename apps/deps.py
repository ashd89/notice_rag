# app/deps.py
"""
FastAPI에서 재사용할 공용 의존성 모듈 lazy singleton 초기화
"""
from __future__ import annotations
import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
# from openai import AsyncOpenAI
from pydantic_settings import BaseSettings

# from langchain.embeddings import OpenAIEmbeddings
load_dotenv()

# ---------- Settings ----------
class Settings(BaseSettings):
  # API Keys
  openai_api_key: str
  gemini_api_key: str

  # Vector
  embed_provider: str = "tei"
  embed_model: str = "BAAI/bge-m3"
  embed_dim: int = 1024
  embeddings_base_url: str = "http://localhost:8080/v1"
  embeddings_api_key: str = "dummy"    

  # PGVector
  pg_conn: str
  collection_name: str = "uos_announcement"
  embed_model: str = "text-embedding-3-small"
  use_jsonb: bool = True

  # LLM
  chat_model: str = "gpt-4o-mini"        # 최종 응답 생성용
  small_model: str = "gpt-4o-mini"       # 가벼운 재작성/가드/검증용
  temperature: float = 0.0
  request_timeout: int = 60              # seconds

  # OCR
  ocr_timeout: float = 10.0
  azure_docint_endpoint: Optional[str] = None
  azure_docint_key: Optional[str] = None
  ocr_locale_default: Optional[str] = "ko"

  # Retriever 기본값
  retriever_k: int = 6
  retriever_fetch_k: int = 40
  retriever_mmr: bool = False

  class Config:
    env_file = ".env"

# ---------- Settings Singleton ----------
@lru_cache(maxsize=1)
def get_settings() -> Settings:
  return Settings()

# # ---------- Embeddings ----------
# @lru_cache(maxsize=1)
# def get_embeddings() -> OpenAIEmbeddings:
#   cfg = get_settings() 
#   if cfg.embed_provider == "tei":
#     return OpenAIEmbeddings(
#       model=cfg.embed_model,
#       api_key=cfg.embeddings_api_key,
#       base_url=cfg.embeddings_base_url,
#     )
#   else:
#     return OpenAIEmbeddings(
#       model=cfg.embed_model,
#       api_key=cfg.openai_api_key,
#     )