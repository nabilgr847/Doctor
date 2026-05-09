import os, re, json, time
from datetime import datetime, timedelta
from agents.search_team import search_all_unrestricted, search_serpapi
from agents.medical_team import query_all_medical_apis
import pypdf
from groq import Groq

def ask_groq(text):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("❌ GROQ_API_KEY missing")
        return ""
    client = Groq(api_key=key)
    models = ["openai/gpt-oss-120b", "llama-3.1-8b-instant", "llama3-8b-8192"]
    for model in models:
        for attempt in range(2):
            try:
                chat = client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": f"Generate exactly 30 concise medical question-answer pairs. Keep each question under 30 words and each answer under 50 words. Use format:\nQuestion: ...\nAnswer: ...\n\nText:\n{text[:3500]}"
                    }],
                    temperature=0.7,
                    max_tokens=4096
                )
                raw = chat.choices[0].message.content
                print(f"✅ Groq success with {model}, preview: {raw[:200]}")
                return raw
            except Exception as e:
                print(f"❌ Groq {model} error: {e}")
                time.sleep(5)
    return ""

def parse_qa_text(raw):
    if not raw: return []
    matches = re.findall(r'(?:Question|Q):\s*(.*?)\n\s*(?:Answer|A):\s*(.*?)(?=\n\s*(?:Question|Q):|$)', raw, re.DOTALL | re.IGNORECASE)
    qa = [{"question": q.strip(), "answer": a.strip()} for q, a in matches]
    if qa: return qa
    matches2 = re.findall(r'\d+\.\s*(?:Question|Q):\s*(.*?)\n\s*(?:Answer|A):\s*(.*?)(?=\n\d+\.\s*(?:Question|Q):|$)', raw, re.DOTALL | re.IGNORECASE)
    return [{"question": q.strip(), "answer": a.strip()} for q, a in matches2]

def process_uploaded_books():
    book_text = ""
    folder = "upload_books"
    if not os.path.exists(folder): return book_text
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder, filename)
            try:
                reader = pypdf.PdfReader(filepath)
                pages = [p.extract_text() for p in reader.pages[:10] if p.extract_text()]
                book_text += f"\n--- {filename} ---\n" + "\n".join(pages)
                processed = os.path.join(folder, "processed")
                os.makedirs(processed, exist_ok=True)
                os.replace(filepath, os.path.join(processed, filename))
            except Exception as e:
                print(f"PDF error {filename}: {e}")
    return book_text

def get_output_file():
    base, ext = "dataset", ".jsonl"
    num = 1
    fname = f"{base}{ext}"
    while os.path.exists(fname):
        if os.path.getsize(fname) < 500 * 1024 * 1024:
            return fname
        num += 1
        fname = f"{base}_{num}{ext}"
    return fname

def main():
    print(f"🚀 Doctor Non-Stop Run started @ {datetime.now()}")
    end_time = datetime.utcnow() + timedelta(hours=5, minutes=50)
    
    while datetime.utcnow() < end_time:
        start_cycle = time.time()
        
        # 1. তথ্য সংগ্রহ
        book = process_uploaded_books()
        search_data = search_all_unrestricted()
        medical_data = query_all_medical_apis()
        hour = datetime.utcnow().hour
        serp = search_serpapi() if hour in [0,8,16] else ""
        combined = book + "\n" + search_data + "\n" + medical_data + "\n" + serp
        print(f"📊 Data length: {len(combined)}")
        
        # 2. Groq কল (30 Q&A)
        raw = ask_groq(combined)
        entries = parse_qa_text(raw)
        print(f"📝 {len(entries)} entries extracted")
        
        # 3. ফাইলে লেখা ও পুশ (সিনট্যাক্স এরর ফিক্স করা হয়েছে)
        if entries:
            out_file = get_output_file()
            with open(out_file, "a", encoding="utf-8") as f:
                for e in entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
            os.system("git config user.name 'God-Doctor-Bot'")
            os.system("git config user.email 'bot@doctor.ai'")
            os.system(f"git add {out_file}")
            os.system(f"git commit -m 'Auto-update dataset {timestamp}' || echo 'No changes'")
            os.system("git push")
            print(f"✅ {len(entries)} entries pushed to repo")
        
        # 4. অপেক্ষা
        elapsed = time.time() - start_cycle
        sleep_time = max(10, 15 - elapsed)
        print(f"⏳ Sleeping {sleep_time:.1f}s...")
        time.sleep(sleep_time)
    
    print("🏁 Non-stop run completed. Next scheduled run will take over.")

if __name__ == "__main__":
    main()
