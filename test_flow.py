import requests

BASE_URL = "http://127.0.0.1:8000"

def run_test_scenario():
    print("🚀 BẮT ĐẦU TEST LUỒNG LMS (RAG + SCORING)\n")

    payload_gen = {
        "topic": "Python Optimization",
        "level": "Intermediate"
    }

    resp_gen = requests.post(f"{BASE_URL}/quiz/generate", json=payload_gen)
    resp_gen.raise_for_status()
    data_gen = resp_gen.json()
    quiz_id = data_gen['quiz_id']
    questions = data_gen['questions']

    print(f"Created quiz {quiz_id}, questions: {len(questions)}")
    # Build user answers: keys MUST be strings (JSON object keys are strings)
    user_answers = {}
    for idx, q in enumerate(questions):
        q_id = str(q['question_id'])
        user_answers[q_id] = "B" if idx < 3 else "A"

    payload_submit = {"answers": user_answers}
    resp_submit = requests.post(f"{BASE_URL}/quiz/{quiz_id}/submit", json=payload_submit)
    resp_submit.raise_for_status()
    print("Submit result:", resp_submit.json())

if __name__ == "__main__":
    run_test_scenario()
