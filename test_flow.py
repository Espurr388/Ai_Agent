import requests

# Cấu hình
BASE_URL = "http://127.0.0.1:8000"

def run_test_scenario():
    print(" BẮT ĐẦU TEST LUỒNG LMS (RAG + SCORING)\n")

    # --- BƯỚC 1: GENERATE ---
    print(f"[Step 1] Đang gọi API /quiz/generate...")
    payload_gen = {
        "topic": "Python Optimization",
        "level": "Intermediate"
    }
    
    try:
        resp_gen = requests.post(f"{BASE_URL}/quiz/generate", json=payload_gen)
        resp_gen.raise_for_status() # Báo lỗi nếu code != 200
        data_gen = resp_gen.json()
        
        quiz_id = data_gen['quiz_id']
        questions = data_gen['questions']
        
        print(f"  ✅ Đã tạo Quiz ID: {quiz_id}")
        print(f"  ✅ Số lượng câu hỏi: {len(questions)}")
        print(f"  ✅ ID câu hỏi đầu tiên: {questions[0]['question_id']}")
        print(f"  ✅ Nguồn (Source) câu 1: {questions[0]['sources']}")
        
    except Exception as e:
        print(f"  ❌ Lỗi Bước 1: {e}")
        return

    print("-" * 50)

    # --- BƯỚC 2: USER ANSWERS ---
    print(f"[Step 2] Đang giả lập người dùng trả lời...")
    
    # Chiến thuật test: 
    # - 3 câu đầu chọn "B" (Đáp án đúng theo mock logic cũ)
    # - 2 câu sau chọn "A" (Đáp án sai)
    # -> Kỳ vọng: Đúng 3/5 -> 60%
    
    user_answers = {}
    for index, q in enumerate(questions):
        q_id = q['question_id']
        if index < 3:
            user_answers[q_id] = "B" # Giả sử B là đúng
        else:
            user_answers[q_id] = "A" # Giả sử A là sai
            
    print(f"  User gửi đáp án: {user_answers}")
    print("-" * 50)

    # --- BƯỚC 3: NỘP BÀI & CHẤM ĐIỂM (SUBMIT) ---
    print(f"[Step 3] Đang gọi API /quiz/{quiz_id}/submit...")
    
    payload_submit = {
        "answers": user_answers
    }
    
    try:
        resp_submit = requests.post(f"{BASE_URL}/quiz/{quiz_id}/submit", json=payload_submit)
        resp_submit.raise_for_status()
        result = resp_submit.json()
        
        print(f"   Kết quả trả về từ Server:")
        print(f"     - Quiz ID: {result['quiz_id']}")
        print(f"     - Score: {result['score']}/{result['total']}")
        print(f"     - Percentage: {result['percentage']}%")
        print(f"     - Detail: {result['detail']}")
        
        # --- BƯỚC 4: KIỂM TRA KẾT QUẢ (ASSERTION) ---
        print("-" * 50)
        print("[Step 4] Tự động kiểm tra logic...")
        
        if result['score'] == 3 and result['percentage'] == 60.0:
            print("   TEST PASSED: Điểm số tính toán chính xác (3/5)!")
        else:
            print(f"   TEST WARNING: Điểm số không khớp kỳ vọng. (Kỳ vọng 3, Thực tế {result['score']})")
            print("  (Kiểm tra lại xem Mock LLM có đang random đáp án không?)")
            
    except Exception as e:
        print(f"   Lỗi Bước 3: {e}")
        print(resp_submit.text)

if __name__ == "__main__":

    run_test_scenario()
