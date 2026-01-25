-- Schema mẫu cho PostgreSQL
-- Chạy once để tạo database structure
CREATE TABLE IF NOT EXISTS quizzes (
  id SERIAL PRIMARY KEY,
  topic TEXT NOT NULL,
  difficulty_level TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE TABLE IF NOT EXISTS questions (
  id SERIAL PRIMARY KEY,
  quiz_id INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  question_type TEXT NOT NULL,
  options JSONB NOT NULL,        -- {"A":"..","B":"..","C":"..","D":".."}
  correct_answer TEXT NOT NULL, -- "A"/"B"/"C"/"D"
  explanation TEXT,
  sources JSONB,                -- ["101","102"]
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);