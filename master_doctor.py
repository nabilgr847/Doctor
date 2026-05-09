import os, re, json, time
from datetime import datetime, timedelta
from agents.search_team import search_all_unrestricted, search_serpapi
from agents.medical_team import query_all_medical_apis
import pypdf
from groq import Groq

# ---------- Groq (প্রথম ব্যাকআপ মডেলসহ) ----------
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
                        "content": f"Generate exactly 50 concise medical question-answer pairs. Keep each question under 30 words and each answer under 50 words. Use format:\nQuestion: ...\nAnswer: ...\n\nText:\n{text[:3500]}"
                    }],
                    temperature=0.7,
                    max_tokens=6144  # 50 জোড়ার জন্য পর্যাপ্ত
                )
                raw = chat.choices[0].message.content
                print(f"✅ Groq success with {model}, preview: {raw[:200]}")
                return raw
            except Exception as e:
                print(f"❌ Groq {model} error: {e}")
                time.sleep(5)
    return ""

# ---------- Pollinations.AI (সম্পূর্ণ ফ্রি, কোনো Key নয়) ----------
def ask_pollinations(text):
    try:
        r = requests.post("https://text.pollinations.ai/openai/v1/chat/completions", json={
            "messages": [{
                "role": "user",
                "content": f"Generate exactly 50 concise medical question-answer pairs. Keep each question under 30 words and each answer under 50 words. Use format:\nQuestion: ...\nAnswer: ...\n\nText:\n{text[:3500]}"
            }],
            "model": "openai",
            "temperature": 0.7
        })
        if r.status_code == 200:
            raw = r.json()["choices"][0]["message"]["content"]
            print(f"✅ Pollinations success, preview: {raw[:200]}")
            return raw
    except Exception as e:
        print(f"❌ Pollinations error: {e}")
    return ""

# ---------- উন্নত Q&A পার্সার (সংখ্যাযুক্ত ও বোল্ড ফরম্যাট ধরবে) ----------
def parse_qa_text(raw):
    if not raw:
        return []
    # প্রধান প্যাটার্ন: Question/Answer বা Q/A (ঐচ্ছিক নম্বর সহ)
    matches = re.findall(r'\d*\.?\s*(?:Question|Q):\s*(.*?)\n\s*(?:Answer|A):\s*(.*?)(?=\n\s*\d*\.?\s*(?:Question|Q):|$)', raw, re.DOTALL | re.IGNORECASE)
    qa = [{"question": q.strip(), "answer": a.strip()} for q, a in matches]
    if qa:
        return qa
    # ব্যাকআপ: **Question:** বা **Answer:**
    matches2 = re.findall(r'\*?\*?(?:Question|Q)\*?\*?:\s*(.*?)\n\s*\*?\*?(?:Answer|A)\*?\*?:\s*(.*?)(?=\n\s*\*?\*?(?:Question|Q)|$)', raw, re.DOTALL | re.IGNORECASE)
    qa2 = [{"question": q.strip(), "answer": a.strip()} for q, a in matches2]
    return qa2

# ---------- পিডিএফ প্রসেসিং ----------
def process_uploaded_books():
    book_text = ""
    folder = "upload_books"
    if not os.path.exists(folder):
        return book_text
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

# ---------- ফাইল সাইজ ব্যবস্থাপনা ----------
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

# ---------- প্রধান নন-স্টপ লুপ ----------
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
        
        # 2. Groq + Pollinations (সমান্তরালে নয়, পর্যায়ক্রমে)
        raw_groq = ask_groq(combined)
        raw_poll = ask_pollinations(combined)
        
        entries_groq = parse_qa_text(raw_groq)
        entries_poll = parse_qa_text(raw_poll)
        
        all_entries = entries_groq + entries_poll
        print(f"📝 Groq: {len(entries_groq)}, Pollinations: {len(entries_poll)} → Total: {len(all_entries)}")
        
        # 3. ফাইলে লেখা ও পুশ
        if all_entries:
            out_file = get_output_file()
            with open(out_file, "a", encoding="utf-8") as f:
                for e in all_entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
            os.system("git config user.name 'God-Doctor-Bot'")
            os.system("git config user.email 'bot@doctor.ai'")
            os.system(f"git add {out_file}")
            os.system(f"git commit -m 'Auto-update dataset {timestamp}' || echo 'No changes'")
            
            # টোকেন সেট করে পুশ
            token = os.environ["GH_TOKEN"]
            repo = os.environ["REPOSITORY"]
            remote_url = f"https://x-access-token:{token}@github.com/{repo}.git"
            os.system(f"git remote set-url origin {remote_url}")
            os.system("git push")
            
            print(f"✅ {len(all_entries)} entries pushed to repo")
        else:
            print("⚠️ No entries this cycle.")
        
        # 4. বিরতি
        elapsed = time.time() - start_cycle
        sleep_time = max(10, 15 - elapsed)
        print(f"⏳ Sleeping {sleep_time:.1f}s...")
        time.sleep(sleep_time)
    
    print("🏁 Non-stop run completed. Next scheduled run will take over.")

if __name__ == "__main__":
    main()
