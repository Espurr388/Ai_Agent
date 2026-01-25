import json
import time
import os
import requests # Cần cài: pip install requests 
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# --- 1. CẤU HÌNH ---
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "lms_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

app = FastAPI()

# --- 2. ĐỊNH NGHĨA DTO (DATA TRANSFER OBJECTS) ---

class QuizRequest(BaseModel):
    topic: str
    level: str 

    @validator('level')
    def validate_level(cls, v):
        allowed = ["Beginner", "Intermediate", "Advanced"]
        if v not in allowed:
            raise ValueError(f"Level không hợp lệ. Chỉ chấp nhận: {allowed}")
        return v

class QuestionDTO(BaseModel):
    question_id: Optional[int] = None # Để FE biết ID khi submit
    content: str
    question_type: str 
    options: dict           # {"A": "...", "B": "..."}
    correct_answer: str     # "A", "B", "C", hoặc "D"
    explanation: str        # MỚI: Giải thích
    sources: List[str]      # MỚI: Danh sách chunk_id

class QuizResponse(BaseModel):
    quiz_id: int
    topic: str
    level: str
    status: str
    questions: List[QuestionDTO]

# DTO cho việc Submit (Chấm điểm)
class SubmitRequest(BaseModel):
    answers: Dict[int, str] # {question_id: "A", question_id_2: "C"}

class SubmitResponse(BaseModel):
    quiz_id: int
    score: int
    total: int
    percentage: float
    detail: Dict[str, str] # {question_id: "Correct/Incorrect"}

# --- 3. LOGIC RAG & PROMPT ---

def call_rag_search(topic: str) -> str:
    """
    Giả lập gọi API /rag/search để lấy context chunks.
    Trong thực tế, bạn sẽ dùng requests.post('http://rag-service/search', ...)
    """
    print(f"[RAG] Đang tìm kiếm tài liệu cho topic: {topic}...")
    # Giả lập context trả về từ vector DB
    return f"""
    [Chunk 101] {topic} cơ bản bao gồm các khái niệm định nghĩa và cú pháp.
    [Chunk 102] {topic} nâng cao tập trung vào tối ưu hiệu năng và bảo mật.
    [Chunk 103] Một số lỗi thường gặp trong {topic} là quản lý bộ nhớ sai.
    """

def get_rag_prompt_template(topic: str, level: str, context: str) -> str:
    return f"""
    Role: Expert Exam Creator.
    Task: Create 5 Multiple Choice Questions (MCQ) about "{topic}" at "{level}" level.
    Context Information:
    {context}
    
    Requirements:
    1. 4 options (A, B, C, D) per question.
    2. Indicate the correct answer (Key).
    3. Provide a short explanation based on the context.
    4. Cite the source chunk IDs (e.g., Chunk 101).
    5. Output JSON format only.
    """

# --- 4. GIẢ LẬP LLM (STUB) ---

def mock_llm_rag_generate(prompt: str):
    print(f"\n[LLM] Đang sinh câu hỏi với Context...\n")
    time.sleep(1) 

    # Giả lập trả về 5 câu hỏi MCQ chuẩn cấu trúc
    # Lưu ý: Question Type mặc định là "MCQ"
    mock_questions = []
    for i in range(1, 6):
        mock_questions.append({
            "content": f"Câu hỏi số {i} dựa trên context về {prompt[:10]}...?",
            "question_type": "MCQ",
            "options": {
                "A": f"Phương án A câu {i}",
                "B": f"Phương án B câu {i} (Đúng)",
                "C": f"Phương án C câu {i}",
                "D": f"Phương án D câu {i}"
            },
            "correct_answer": "B",
            "explanation": f"Vì context Chunk 10{i%3 + 1} nói rằng...",
            "sources": [f"Chunk 10{i%3 + 1}"]
        })
    return mock_questions

# --- 5. API ENDPOINTS ---

@app.post("/quiz/generate", response_model=QuizResponse)
def generate_quiz_endpoint(request: QuizRequest):
    conn = None
    try:
        # BƯỚC A: RAG Search (Lấy context)
        context_data = call_rag_search(request.topic)
        
        # BƯỚC B: Tạo Prompt & Gọi LLM
        prompt_text = get_rag_prompt_template(request.topic, request.level, context_data)
        generated_questions = mock_llm_rag_generate(prompt_text)
        
        # BƯỚC C: Lưu Database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # C1. Insert Quiz
        cur.execute(
            "INSERT INTO quizzes (topic, difficulty_level) VALUES (%s, %s) RETURNING id",
            (request.topic, request.level)
        )
        new_quiz_id = cur.fetchone()[0]
        
        response_questions = []
        
        # C2. Insert Questions (5 câu)
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
                    q['content'], 
                    q['question_type'], 
                    json.dumps(q['options']), 
                    q['correct_answer'],
                    q['explanation'],
                    json.dumps(q['sources']) # Lưu list sources dưới dạng JSON
                )
            )
            q_id = cur.fetchone()[0]
            
            # Map dữ liệu để trả về Client (gắn thêm ID vừa tạo)
            q_dto = QuestionDTO(question_id=q_id, **q)
            response_questions.append(q_dto)
            
        conn.commit()
        
        return {
            "quiz_id": new_quiz_id,
            "topic": request.topic,
            "level": request.level,
            "status": "Success",
            "questions": response_questions
        }

    except Exception as e:
        if conn: conn.rollback()
        print(f"Lỗi Generate: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn: conn.close()

@app.post("/quiz/{quiz_id}/submit", response_model=SubmitResponse)
def submit_quiz_endpoint(quiz_id: int, submit_data: SubmitRequest):
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # 1. Lấy đáp án đúng từ DB
        cur.execute("SELECT id, correct_answer FROM questions WHERE quiz_id = %s", (quiz_id,))
        db_questions = cur.fetchall()
        
        if not db_questions:
            raise HTTPException(status_code=404, detail="Quiz not found or no questions")

        # 2. Chấm điểm
        score = 0
        total = len(db_questions)
        detail_result = {}

        # Tạo map id -> correct_answer để tra cứu nhanh
        correct_map = {q['id']: q['correct_answer'] for q in db_questions}

        for q_id, user_ans in submit_data.answers.items():
            # Chuyển q_id từ string (nếu JSON gửi lên là string key) sang int
            q_id = int(q_id) 
            
            if q_id in correct_map:
                is_correct = (user_ans.strip().upper() == correct_map[q_id].strip().upper())
                if is_correct:
                    score += 1
                    detail_result[str(q_id)] = "Correct"
                else:
                    detail_result[str(q_id)] = "Incorrect"
            else:
                detail_result[str(q_id)] = "Question ID not in this quiz"

        percentage = (score / total) * 100 if total > 0 else 0

        return {
            "quiz_id": quiz_id,
            "score": score,
            "total": total,
            "percentage": round(percentage, 2),
            "detail": detail_result
        }

    except Exception as e:
        print(f"Lỗi Submit: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn: conn.close()
