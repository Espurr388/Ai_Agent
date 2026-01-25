import os
import json
import time
import logging
from typing import List, Optional, Dict, Tuple, Any

import requests  # pip install requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from dotenv import load_dotenv

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

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
    options: Dict[str, str]
    correct_answer: str
    explanation: str
    sources: List[str]

class QuizResponse(BaseModel):
    quiz_id: int
    topic: str
    level: str
    status: str
    questions: List[QuestionDTO]

class SubmitRequest(BaseModel):
    answers: Dict[str, str]  # use string keys to be explicit

class SubmitResponse(BaseModel):
    quiz_id: int
    score: int
    total: int
    percentage: float
    detail: Dict[str, str]

# --- RAG integration ---
def call_rag_search(topic: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Gọi service /rag/search (nếu có). Kết quả mong đợi là danh sách chunks:
    [{ "id": 101, "text": "..." }, ...] or [{ "chunk_id": "101", "text": "..." }, ...]
    Trả về (context_text, chunks_list).
    Nếu lỗi, trả về fallback stub.
    """
    logger.info(f"[RAG] Searching for topic: {topic}")
    try:
        resp = requests.post(RAG_SEARCH_URL, json={"query": topic}, timeout=RAG_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise ValueError("RAG response is not a list")
        # Normalize chunks to have id and text
        chunks = []
        for c in data:
            cid = c.get("id") or c.get("chunk_id") or c.get("chunkId") or c.get("chunk")
            txt = c.get("text") or c.get("content") or c.get("chunk_text")
            if cid is None or txt is None:
                # skip invalid chunk
                continue
            chunks.append({"id": str(cid), "text": str(txt)})
        if not chunks:
            raise ValueError("No valid chunks returned")
        context = "\n".join([f"[Chunk {c['id']}] {c['text']}" for c in chunks])
        return context, chunks
    except Exception as e:
        logger.warning(f"[RAG] external call failed: {e}. Using fallback stub.")
        # Fallback stub (keeps same format)
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
Task: Create 5 Multiple Choice Questions (MCQ) about "{topic}" at "{level}" level.
Context Information:
{context}

Requirements:
1. Output exactly 5 questions.
2. Each question must have 4 options (A, B, C, D).
3. Indicate the correct answer using the "correct_answer" field ("A"/"B"/"C"/"D").
4. Provide a short explanation in "explanation".
5. Provide "sources": an array of chunk ids used (e.g., ["101","102"]).
6. Output JSON array only, e.g.:
[
  {{
    "content":"..",
    "question_type":"MCQ",
    "options":{{"A":"..","B":"..","C":"..","D":".."}},
    "correct_answer":"B",
    "explanation":"..",
    "sources":["101"]
  }},
  ...
]
"""

# --- Mock LLM (stub) ---
def mock_llm_rag_generate(prompt: str, chunks: List[Dict[str, str]]):
    logger.info("[LLM] generating (mock) questions...")
    time.sleep(0.5)
    mock_questions = []
    for i in range(1, 6):
        chunk_id = chunks[(i - 1) % len(chunks)]["id"]
        mock_questions.append({
            "content": f"Câu hỏi số {i} về {prompt[:30]}",
            "question_type": "MCQ",
            "options": {
                "A": f"Phương án A câu {i}",
                "B": f"Phương án B câu {i}",  # in mock, we'll mark B as correct
                "C": f"Phương án C câu {i}",
                "D": f"Phương án D câu {i}"
            },
            "correct_answer": "B",
            "explanation": f"Dựa trên Chunk {chunk_id} nội dung ...",
            "sources": [chunk_id]
        })
    return mock_questions

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
        # 1) RAG search
        context_text, chunks = call_rag_search(request.topic)

        # 2) Build prompt + call LLM (mock here)
        prompt_text = get_rag_prompt_template(request.topic, request.level, context_text)
        generated_questions = mock_llm_rag_generate(prompt_text, chunks)

        # Validate generated format (basic)
        if not isinstance(generated_questions, list) or len(generated_questions) != 5:
            raise ValueError("LLM did not return 5 questions in expected format")

        # 3) Persist to DB using pool
        conn = get_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO quizzes (topic, difficulty_level) VALUES (%s, %s) RETURNING id",
                (request.topic, request.level),
            )
            new_quiz_id = cur.fetchone()[0]

            response_questions = []
            for q in generated_questions:
                cur.execute(
                    """
                    INSERT INTO questions
                    (quiz_id, content, question_type, options, correct_answer, explanation, sources)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        new_quiz_id,
                        q["content"],
                        q["question_type"],
                        json.dumps(q["options"]),
                        q["correct_answer"],
                        q["explanation"],
                        json.dumps(q["sources"]),
                    ),
                )
                q_id = cur.fetchone()[0]
                q_dto = QuestionDTO(question_id=q_id, **q)
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

@app.post("/quiz/{quiz_id}/submit", response_model=SubmitResponse)
def submit_quiz_endpoint(quiz_id: int, submit_data: SubmitRequest):
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, correct_answer FROM questions WHERE quiz_id = %s", (quiz_id,))
        db_questions = cur.fetchall()
        if not db_questions:
            raise HTTPException(status_code=404, detail="Quiz not found or no questions")

        # map id -> correct_answer
        correct_map = {q["id"]: q["correct_answer"] for q in db_questions}
        total = len(db_questions)
        score = 0
        detail_result: Dict[str, str] = {}

        for q_id_str, user_ans in submit_data.answers.items():
            try:
                q_id = int(q_id_str)
            except Exception:
                detail_result[str(q_id_str)] = "Invalid question id format"
                continue
            if q_id in correct_map:
                is_correct = (str(user_ans).strip().upper() == str(correct_map[q_id]).strip().upper())
                if is_correct:
                    score += 1
                    detail_result[str(q_id)] = "Correct"
                else:
                    detail_result[str(q_id)] = "Incorrect"
            else:
                detail_result[str(q_id)] = "Question ID not in this quiz"

        percentage = (score / total) * 100 if total > 0 else 0.0
        return {
            "quiz_id": quiz_id,
            "score": score,
            "total": total,
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
