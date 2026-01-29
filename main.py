# contents of main.py
# (cập nhật từ phiên bản bạn gửi; bổ sung OpenAI/embeddings, chunk persistence, endpoints)
import os
import json
import time
import logging
from typing import List, Optional, Dict, Tuple, Any

import requests  # pip install requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, validator
from dotenv import load_dotenv

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

# Optional OpenAI integration
try:
    import openai
    import numpy as np
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

load_dotenv()

# --- Config ---
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "lms_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

RAG_SEARCH_URL = os.getenv("RAG_SEARCH_URL", "http://localhost:9000/rag/search")
RAG_TIMEOUT = float(os.getenv("RAG_TIMEOUT", "3.0"))

POOL_MIN = int(os.getenv("DB_POOL_MIN", "1"))
POOL_MAX = int(os.getenv("DB_POOL_MAX", "10"))

# OpenAI config
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() in ("1", "true", "yes")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # example
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

if USE_OPENAI:
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK (openai) or numpy not installed. Install them or disable USE_OPENAI.")
    if not OPENAI_API_KEY:
        raise RuntimeError("ENABLE OPENAI: set OPENAI_API_KEY in env")
    openai.api_key = OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lms_app")

app = FastAPI()
db_pool: Optional[SimpleConnectionPool] = None

# --- DTOs ---
class QuizRequest(BaseModel):
    topic: str
    level: str

    @validator("level")
    def validate_level(cls, v):
        allowed = ["Beginner", "Intermediate", "Advanced"]
        if v not in allowed:
            raise ValueError(f"Level không hợp lệ. Chỉ chấp nhận: {allowed}")
        return v

class QuestionDTO(BaseModel):
    question_id: Optional[int] = None
    content: str
    question_type: str
    options: Optional[Dict[str, str]] = None
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None
    sources: Optional[List[str]] = None
    rubric: Optional[Dict[str, Any]] = None    # for Essay
    max_score: int = 1

class QuizResponse(BaseModel):
    quiz_id: int
    topic: str
    level: str
    status: str
    questions: List[QuestionDTO]

class SubmitRequest(BaseModel):
    answers: Dict[str, str]  # keys are question_id as strings

class SubmitResponse(BaseModel):
    quiz_id: int
    score: float
    correct_count: int
    total_points: float
    percentage: float
    detail: Dict[str, Dict[str, Any]]

# --- RAG integration ---
def call_rag_search(topic: str) -> Tuple[str, List[Dict[str, Any]]]:
    logger.info(f"[RAG] Searching for topic: {topic}")
    try:
        resp = requests.post(RAG_SEARCH_URL, json={"query": topic}, timeout=RAG_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise ValueError("RAG response is not a list")
        chunks = []
        for c in data:
            cid = c.get("id") or c.get("chunk_id") or c.get("chunkId") or c.get("chunk")
            txt = c.get("text") or c.get("content") or c.get("chunk_text")
            if cid is None or txt is None:
                continue
            chunks.append({"id": str(cid), "text": str(txt)})
        if not chunks:
            raise ValueError("No valid chunks returned")
        context = "\n".join([f"[Chunk {c['id']}] {c['text']}" for c in chunks])
        return context, chunks
    except Exception as e:
        logger.warning(f"[RAG] external call failed: {e}. Using fallback stub.")
        fallback_chunks = [
            {"id": "101", "text": f"{topic} cơ bản: định nghĩa và cú pháp."},
            {"id": "102", "text": f"{topic} tối ưu: performance và best practices."},
            {"id": "103", "text": f"{topic} lỗi thường gặp: memory, concurrency."},
        ]
        context = "\n".join([f"[Chunk {c['id']}] {c['text']}" for c in fallback_chunks])
        return context, fallback_chunks

def get_rag_prompt_template(topic: str, level: str, context: str) -> str:
    return f"""
Role: Expert Exam Creator.
Task: Create 5 questions about "{topic}" at "{level}" level.
Context Information:
{context}

Requirements:
1. Output exactly 5 questions.
2. At least 4 MCQ (A,B,C,D) and up to 1 Essay.
3. MCQ: options, correct_answer ("A"/"B"/"C"/"D"), explanation, sources (chunk ids).
4. Essay: rubric {{"keywords": [...], "notes":"..."}} and max_score integer, and sources.
5. Output JSON array only.
"""

# --- LLM generation (mock + OpenAI) ---
def mock_llm_rag_generate(prompt: str, chunks: List[Dict[str, str]]):
    logger.info("[LLM MOCK] generating mock questions...")
    time.sleep(0.5)
    mock_questions = []
    for i in range(1, 6):
        chunk_id = chunks[(i - 1) % len(chunks)]["id"]
        if i < 5:
            mock_questions.append({
                "content": f"Câu hỏi số {i} về {prompt[:30]}",
                "question_type": "MCQ",
                "options": {"A": f"A{i}", "B": f"B{i}", "C": f"C{i}", "D": f"D{i}"},
                "correct_answer": "B",
                "explanation": f"Dựa trên Chunk {chunk_id} ...",
                "sources": [chunk_id],
                "max_score": 1
            })
        else:
            keywords = [f"keyword_{chunk_id}_1", f"keyword_{chunk_id}_2"]
            mock_questions.append({
                "content": f"Viết về {prompt[:30]} (Essay)",
                "question_type": "Essay",
                "rubric": {"keywords": keywords, "notes": "Nêu định nghĩa và 2 điểm tối ưu."},
                "max_score": 5,
                "explanation": f"Baseline: nhắc tới {', '.join(keywords)}",
                "sources": [chunk_id]
            })
    return mock_questions

def openai_generate(prompt: str, chunks: List[Dict[str, str]]):
    """
    Example: use OpenAI to generate JSON. This is an illustrative snippet; in production
    you must handle token limits, streaming, robust parsing, retries, and safety.
    """
    logger.info("[LLM OPENAI] calling OpenAI for generation...")
    system_message = "You are an expert exam creator that outputs JSON array only."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    resp = openai.ChatCompletion.create(model=OPENAI_MODEL, messages=messages, max_tokens=1200, temperature=0.2)
    text = resp.choices[0].message.content
    # We expect the model to return a JSON array. Parse safely.
    # Basic attempt: find first '[' and last ']' and json.loads slice.
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        json_text = text[start:end]
        data = json.loads(json_text)
        return data
    except Exception as e:
        logger.exception("Failed to parse OpenAI output; falling back to mock")
        return mock_llm_rag_generate(prompt, chunks)

def generate_questions_with_llm(prompt: str, chunks: List[Dict[str, str]]):
    if USE_OPENAI:
        return openai_generate(prompt, chunks)
    else:
        return mock_llm_rag_generate(prompt, chunks)

# --- helper: embeddings & semantic grading ---
def embedding_for_texts(texts: List[str]) -> List[List[float]]:
    if not USE_OPENAI:
        raise RuntimeError("Embeddings requested but USE_OPENAI is false")
    resp = openai.Embedding.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
    return [r["embedding"] for r in resp["data"]]

def cosine_sim(a: List[float], b: List[float]) -> float:
    aa = np.array(a, dtype=float)
    bb = np.array(b, dtype=float)
    if np.linalg.norm(aa) == 0 or np.linalg.norm(bb) == 0:
        return 0.0
    return float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb)))

def semantic_score_essay(user_text: str, reference_text: str, max_score: int) -> float:
    """
    Compute similarity-based score in [0, max_score] using embeddings. Returns float.
    """
    try:
        emb_user, emb_ref = embedding_for_texts([user_text, reference_text])
        sim = cosine_sim(emb_user, emb_ref)
        # map similarity [0,1] to score, with some thresholds
        # e.g. sim >=0.85 => full score, sim <=0.2 => 0
        score = 0.0
        if sim >= 0.85:
            score = float(max_score)
        elif sim <= 0.2:
            score = 0.0
        else:
            # linear mapping between 0.2..0.85
            score = ((sim - 0.2) / (0.85 - 0.2)) * max_score
        return round(score, 2)
    except Exception:
        logger.exception("Semantic scoring failed; fallback to 0")
        return 0.0

# --- FastAPI lifecycle events ---
@app.on_event("startup")
def startup():
    global db_pool
    logger.info("Starting app and creating DB pool...")
    db_pool = SimpleConnectionPool(POOL_MIN, POOL_MAX, **DB_CONFIG)

@app.on_event("shutdown")
def shutdown():
    global db_pool
    if db_pool:
        logger.info("Closing DB pool...")
        db_pool.closeall()

# --- Helpers for DB operations ---
def get_conn():
    if not db_pool:
        raise RuntimeError("DB pool not initialized")
    return db_pool.getconn()

def put_conn(conn):
    if db_pool and conn:
        db_pool.putconn(conn)

# --- Endpoints ---
@app.post("/quiz/generate", response_model=QuizResponse)
def generate_quiz_endpoint(request: QuizRequest):
    conn = None
    try:
        context_text, chunks = call_rag_search(request.topic)
        prompt_text = get_rag_prompt_template(request.topic, request.level, context_text)
        generated_questions = generate_questions_with_llm(prompt_text, chunks)

        if not isinstance(generated_questions, list) or len(generated_questions) != 5:
            raise ValueError("LLM did not return 5 questions in expected format")

        conn = get_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO quizzes (topic, difficulty_level) VALUES (%s, %s) RETURNING id",
                (request.topic, request.level),
            )
            new_quiz_id = cur.fetchone()[0]

            # persist chunks into new table (so FE can fetch chunk texts later)
            for c in chunks:
                cur.execute(
                    """
                    INSERT INTO chunks (quiz_id, chunk_id, text)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (quiz_id, chunk_id) DO UPDATE SET text = EXCLUDED.text
                    """,
                    (new_quiz_id, c["id"], c["text"])
                )

            response_questions = []
            for q in generated_questions:
                options = q.get("options")
                correct_answer = q.get("correct_answer")
                rubric = q.get("rubric")
                max_score = int(q.get("max_score", 1))
                cur.execute(
                    """
                    INSERT INTO questions
                    (quiz_id, content, question_type, options, correct_answer, explanation, sources, rubric, max_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        new_quiz_id,
                        q["content"],
                        q["question_type"],
                        json.dumps(options) if options is not None else None,
                        correct_answer,
                        q.get("explanation"),
                        json.dumps(q.get("sources") or []),
                        json.dumps(rubric) if rubric is not None else None,
                        max_score,
                    ),
                )
                q_id = cur.fetchone()[0]
                q_dto = QuestionDTO(
                    question_id=q_id,
                    content=q["content"],
                    question_type=q["question_type"],
                    options=q.get("options"),
                    correct_answer=correct_answer,
                    explanation=q.get("explanation"),
                    sources=q.get("sources"),
                    rubric=rubric,
                    max_score=max_score
                )
                response_questions.append(q_dto)

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

        return {
            "quiz_id": new_quiz_id,
            "topic": request.topic,
            "level": request.level,
            "status": "Success",
            "questions": response_questions,
        }

    except Exception as e:
        logger.exception("Error while generating quiz")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            put_conn(conn)

@app.get("/quiz/{quiz_id}", response_model=QuizResponse)
def get_quiz(quiz_id: int, include_chunks: bool = Query(False, description="Include chunk texts for sources")):
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, topic, difficulty_level FROM quizzes WHERE id = %s", (quiz_id,))
        quiz = cur.fetchone()
        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found")

        cur.execute("""
            SELECT id, content, question_type, options, correct_answer, explanation, sources, rubric, max_score
            FROM questions WHERE quiz_id = %s ORDER BY id
        """, (quiz_id,))
        rows = cur.fetchall()
        questions = []
        for r in rows:
            q = QuestionDTO(
                question_id=r["id"],
                content=r["content"],
                question_type=r["question_type"],
                options=r["options"],
                correct_answer=r["correct_answer"],
                explanation=r["explanation"],
                sources=r["sources"],
                rubric=r["rubric"],
                max_score=r["max_score"] or 1
            )
            questions.append(q)

        result = {
            "quiz_id": quiz_id,
            "topic": quiz["topic"],
            "level": quiz["difficulty_level"],
            "status": "OK",
            "questions": questions
        }

        if include_chunks:
            cur.execute("SELECT chunk_id, text FROM chunks WHERE quiz_id = %s", (quiz_id,))
            chs = cur.fetchall()
            # attach as metadata (not part of response model)
            result["chunks"] = {c["chunk_id"]: c["text"] for c in chs}

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fetching quiz")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            put_conn(conn)

@app.get("/quiz/{quiz_id}/chunks")
def list_chunks_for_quiz(quiz_id: int):
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT chunk_id, text FROM chunks WHERE quiz_id = %s ORDER BY chunk_id", (quiz_id,))
        rows = cur.fetchall()
        return {"quiz_id": quiz_id, "chunks": [{"chunk_id": r["chunk_id"], "text_preview": (r["text"][:300] + "...") if r["text"] and len(r["text"])>300 else r["text"]} for r in rows]}
    except Exception as e:
        logger.exception("Error listing chunks")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            put_conn(conn)

@app.get("/quiz/{quiz_id}/chunks/{chunk_id}")
def get_chunk_text(quiz_id: int, chunk_id: str):
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT chunk_id, text FROM chunks WHERE quiz_id = %s AND chunk_id = %s", (quiz_id, chunk_id))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Chunk not found")
        return {"quiz_id": quiz_id, "chunk_id": row["chunk_id"], "text": row["text"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fetching chunk")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            put_conn(conn)

@app.post("/quiz/{quiz_id}/submit", response_model=SubmitResponse)
def submit_quiz_endpoint(quiz_id: int, submit_data: SubmitRequest):
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT id, question_type, correct_answer, rubric, max_score, explanation, sources
            FROM questions WHERE quiz_id = %s
        """, (quiz_id,))
        db_questions = cur.fetchall()
        if not db_questions:
            raise HTTPException(status_code=404, detail="Quiz not found or no questions")

        q_map = {q["id"]: q for q in db_questions}
        total_points = sum([float(q.get("max_score") or 1) for q in db_questions])

        score = 0.0
        correct_count = 0
        detail_result: Dict[str, Dict[str, Any]] = {}

        # Precompute embeddings for rubric notes for essays if USE_OPENAI
        essay_refs = {}
        if USE_OPENAI:
            essay_texts = []
            essay_qids = []
            for q in db_questions:
                if q["question_type"] == "Essay":
                    rubric = q.get("rubric") or {}
                    ref_text = ""
                    if isinstance(rubric, dict):
                        ref_text = rubric.get("notes") or " ".join(rubric.get("keywords") or [])
                    essay_texts.append(ref_text or "")
                    essay_qids.append(q["id"])
            if essay_texts:
                try:
                    ref_embs = embedding_for_texts(essay_texts)
                    for qid, emb in zip(essay_qids, ref_embs):
                        essay_refs[qid] = emb
                except Exception:
                    logger.exception("Failed to precompute essay reference embeddings; will fallback to keyword matching")
                    essay_refs = {}

        for q_id_str, user_ans in submit_data.answers.items():
            try:
                q_id = int(q_id_str)
            except Exception:
                detail_result[str(q_id_str)] = {
                    "user_answer": user_ans,
                    "awarded_score": 0.0,
                    "max_score": 0,
                    "correct": False,
                    "explanation": "Invalid question id format",
                    "sources": []
                }
                continue

            if q_id not in q_map:
                detail_result[str(q_id)] = {
                    "user_answer": user_ans,
                    "awarded_score": 0.0,
                    "max_score": 0,
                    "correct": False,
                    "explanation": "Question ID not in this quiz",
                    "sources": []
                }
                continue

            q = q_map[q_id]
            q_type = q["question_type"]
            max_score = int(q.get("max_score") or 1)
            explanation = q.get("explanation")
            sources = q.get("sources") or []
            awarded = 0.0
            is_full = False

            if q_type == "MCQ":
                right = str(q.get("correct_answer") or "").strip().upper()
                user_choice = str(user_ans).strip().upper()
                if user_choice == right and right != "":
                    awarded = float(max_score)
                    is_full = True
                else:
                    awarded = 0.0
                detail_result[str(q_id)] = {
                    "user_answer": user_choice,
                    "awarded_score": awarded,
                    "max_score": max_score,
                    "correct": is_full,
                    "explanation": explanation,
                    "sources": sources
                }
            elif q_type == "Essay":
                rubric = q.get("rubric") or {}
                user_text = str(user_ans or "")
                awarded = 0.0
                matched = 0
                if USE_OPENAI and q_id in essay_refs and (isinstance(rubric, dict) and (rubric.get("notes") or rubric.get("keywords"))):
                    # semantic scoring
                    try:
                        ref_emb = essay_refs[q_id]
                        user_emb = embedding_for_texts([user_text])[0]
                        sim = cosine_sim(user_emb, ref_emb)
                        # scoring mapping (same as helper)
                        if sim >= 0.85:
                            awarded = float(max_score)
                        elif sim <= 0.2:
                            awarded = 0.0
                        else:
                            awarded = ((sim - 0.2) / (0.85 - 0.2)) * max_score
                        awarded = round(awarded, 2)
                    except Exception:
                        logger.exception("Semantic essay scoring failed; falling back to keywords")
                        awarded = 0.0
                else:
                    # keyword fallback
                    keywords = rubric.get("keywords") if isinstance(rubric, dict) else None
                    if isinstance(keywords, list) and len(keywords) > 0:
                        lower_user = user_text.lower()
                        for kw in keywords:
                            if kw and kw.lower() in lower_user:
                                matched += 1
                        awarded = (matched / len(keywords)) * float(max_score)
                        awarded = round(awarded, 2)
                    else:
                        awarded = 0.0

                is_full = (isinstance(rubric, dict) and isinstance(rubric.get("keywords"), list) and matched == len(rubric.get("keywords"))) or (USE_OPENAI and awarded == float(max_score))
                detail_result[str(q_id)] = {
                    "user_answer": user_text,
                    "awarded_score": awarded,
                    "max_score": max_score,
                    "correct": is_full,
                    "explanation": explanation or (rubric.get("notes") if isinstance(rubric, dict) else None),
                    "sources": sources,
                    "rubric": rubric
                }
            else:
                detail_result[str(q_id)] = {
                    "user_answer": str(user_ans),
                    "awarded_score": 0.0,
                    "max_score": max_score,
                    "correct": False,
                    "explanation": explanation,
                    "sources": sources
                }

            score += float(awarded)
            if is_full:
                correct_count += 1

        percentage = (score / total_points) * 100 if total_points > 0 else 0.0

        return {
            "quiz_id": quiz_id,
            "score": round(score, 2),
            "correct_count": correct_count,
            "total_points": round(total_points, 2),
            "percentage": round(percentage, 2),
            "detail": detail_result,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error while submitting quiz")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            put_conn(conn)
