# app/deps.py
from __future__ import annotations
import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from langchain.embeddings import OpenAIEmbeddings
from pydantic_settings import BaseSettings

# from langchain.embeddings import OpenAIEmbeddings
load_dotenv()

# ---------- Settings ----------
class Settings(BaseSettings):
  # API Keys
  openai_api_key: str
  gemini_api_key: str

  # PGVector
  db_user: str
  db_password: str
  db_name: str
  db_host: str
  db_port: int
  collection_name: str = "embedding"

  embed_model: str = "text-embedding-3-small"
  embed_dim: int = 1536
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


# ---------- OpenAI Client ----------
_openai_client: Optional[AsyncOpenAI] = None

def get_openai_client() -> AsyncOpenAI:
  """AsyncOpenAI 클라이언트 (싱글톤)."""
  global _openai_client
  if _openai_client is None:
    from openai import AsyncOpenAI
    cfg = get_settings()
    _openai_client = AsyncOpenAI(api_key=cfg.openai_api_key)
  return _openai_client

# ---------- Embeddings ----------
_embeddings: Optional[OpenAIEmbeddings] = None

def get_embeddings() -> OpenAIEmbeddings:
  """OpenAI 임베딩 인스턴스."""
  global _embeddings
  if _embeddings is None:
    cfg = get_settings()
    _embeddings = OpenAIEmbeddings(
        model=cfg.embed_model,
        api_key=cfg.openai_api_key
    )
  return _embeddings
