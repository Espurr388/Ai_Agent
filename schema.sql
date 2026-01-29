-- Schema mẫu cho PostgreSQL (upgrade)
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
  options JSONB,        -- {"A":"..","B":"..","C":"..","D":".."}
  correct_answer TEXT, -- "A"/"B"/"C"/"D" for MCQ
  explanation TEXT,
  sources JSONB,                -- ["101","102"]
  rubric JSONB,                 -- for Essay: {"keywords":["..."], "notes":"..."}
  max_score INTEGER DEFAULT 1,  -- max points for this question
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- New table to persist chunks returned by RAG per quiz
CREATE TABLE IF NOT EXISTS chunks (
  id SERIAL PRIMARY KEY,
  quiz_id INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
  chunk_id TEXT NOT NULL,
  text TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  UNIQUE (quiz_id, chunk_id)
);

-- If updating an existing DB incrementally:
-- ALTER TABLE questions ADD COLUMN IF NOT EXISTS rubric JSONB;
-- ALTER TABLE questions ADD COLUMN IF NOT EXISTS max_score INTEGER DEFAULT 1;
-- CREATE TABLE IF NOT EXISTS chunks (...);
