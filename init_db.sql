-- 1. Users Table (Student Profiles)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    matric_no TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    department TEXT NOT NULL,
    password_hash TEXT NOT NULL
);

-- 2. Syllabus Table (MAT101 Topics)
CREATE TABLE syllabus (
    id SERIAL PRIMARY KEY,
    topic_code TEXT UNIQUE, -- e.g., MAT101.1
    topic_name TEXT NOT NULL -- e.g., "Matrices"
);

-- 3. Questions Table
CREATE TABLE questions (
    id SERIAL PRIMARY KEY,
    topic_id INTEGER REFERENCES syllabus(id),
    content TEXT NOT NULL,
    option_a TEXT, option_b TEXT, option_c TEXT, option_d TEXT,
    correct_answer CHAR(1) -- A, B, C, or D
);

-- 4. Interaction Logs (The AI's Training Data)
CREATE TABLE interaction_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    question_id INTEGER REFERENCES questions(id),
    is_correct INTEGER, -- 1 for True, 0 for False
    response_time FLOAT -- Captured in seconds
);