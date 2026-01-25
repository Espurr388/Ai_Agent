import json
import time
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# --- 1. CẤU HÌNH ---
load_dotenv() # Tự động đọc file .env

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "lms_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "Maiduc2006"), 
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

app = FastAPI()

# --- 2. ĐỊNH NGHĨA DTO ---
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
    content: str
    question_type: str    # Khớp với DB
    options: Optional[dict] = None
    correct_answer: str   # Khớp với DB

class QuizResponse(BaseModel):
    quiz_id: int
    topic: str
    level: str
    status: str
    questions: List[QuestionDTO]

# --- 3. LOGIC PROMPT ---
def get_prompt_template(topic: str, level: str) -> str:
    if level == "Beginner":
        return f"Topic: {topic}. Level: Beginner. Focus: Định nghĩa, nhận biết."
    elif level == "Intermediate":
        return f"Topic: {topic}. Level: Intermediate. Focus: So sánh, phân tích."
    elif level == "Advanced":
        return f"Topic: {topic}. Level: Advanced. Focus: Tối ưu, thiết kế hệ thống."
    return f"Topic: {topic}."

# --- 4. GIẢ LẬP LLM (STUB) ---
def mock_llm_generate(prompt: str, level: str):
    print(f"\n[SYSTEM] Đang gửi Prompt lên LLM: {prompt}\n")
    time.sleep(0.5) 

    # Để khớp với DTO và tên cột trong Database
    if level == "Beginner":
        return [{
            "content": f"Câu hỏi Beginner về {prompt.split('.')[0]}?",
            "question_type": "MCQ",  # <-- Đã sửa
            "options": {"A": "Option 1", "B": "Option 2"},
            "correct_answer": "B"    # <-- Đã sửa
        }]
    elif level == "Intermediate":
        return [{
            "content": f"Câu hỏi Intermediate về {prompt.split('.')[0]}?",
            "question_type": "Essay", # <-- Đã sửa
            "options": {},
            "correct_answer": "Câu trả lời mẫu cho Intermediate." # <-- Đã sửa
        }]
    else: # Advanced
        return [{
            "content": f"Câu hỏi Advanced về {prompt.split('.')[0]}?",
            "question_type": "Essay", # <-- Đã sửa
            "options": {},
            "correct_answer": "Giải pháp tối ưu là..." # <-- Đã sửa
        }]

# --- 5. API ENDPOINT ---
@app.post("/quiz/generate", response_model=QuizResponse)
def generate_quiz_endpoint(request: QuizRequest):
    conn = None
    try:
        # A. Lấy prompt
        prompt_text = get_prompt_template(request.topic, request.level)
        
        # B. Gọi AI giả 
        generated_questions = mock_llm_generate(prompt_text, request.level)
        
        # C. Lưu vào Database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # C1. Insert Quiz
        cur.execute(
            "INSERT INTO quizzes (topic, difficulty_level) VALUES (%s, %s) RETURNING id",
            (request.topic, request.level)
        )
        new_quiz_id = cur.fetchone()[0]
        
        response_data = []
        for q in generated_questions:
            # C2. Insert Questions
            cur.execute(
                """
                INSERT INTO questions (quiz_id, content, question_type, options, correct_answer)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    new_quiz_id, 
                    q['content'], 
                    q['question_type'], 
                    json.dumps(q['options']), 
                    q['correct_answer']
                )
            )
            # Dùng **q để map vào DTO (QuestionDTO cũng đã sửa khớp key)
            response_data.append(QuestionDTO(**q))
            
        conn.commit()
        
        return {
            "quiz_id": new_quiz_id,
            "topic": request.topic,
            "level": request.level,
            "status": "Success",
            "questions": response_data
        }

    except Exception as e:
        if conn: conn.rollback()
        print(f"Lỗi: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn: conn.close()