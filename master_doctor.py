import os, re, json, time
from datetime import datetime
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
    prompt = f"Generate 10 medical Q&A pairs. Use exactly the format:\nQuestion: <question>\nAnswer: <answer>\n\nText:\n{text[:4000]}"
    models = [
        "openai/gpt-oss-120b",          # তোর curl-এ কাজ করেছে
        "llama-3.1-8b-instant",
        "llama3-8b-8192"
    ]
    for model in models:
        for attempt in range(2):
            try:
                chat = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                raw = chat.choices[0].message.content
                print(f"✅ Groq success with {model}")
                print(f"📝 Raw output preview: {raw[:300]}")
                return raw
            except Exception as e:
                print(f"❌ Groq {model} attempt {attempt+1} error: {e}")
                time.sleep(5)
    return ""

def parse_qa_text(raw):
    if not raw: return []
    # 1. "Question: ... Answer: ..." অথবা "Q: ... A: ..."
    matches = re.findall(r'(?:Question|Q):\s*(.*?)\n\s*(?:Answer|A):\s*(.*?)(?=\n\s*(?:Question|Q):|$)', raw, re.DOTALL | re.IGNORECASE)
    qa = [{"question": q.strip(), "answer": a.strip()} for q, a in matches]
    if qa:
        return qa
    # 2. সংখ্যাযুক্ত 1. Question ... Answer ... ধরতে
    matches2 = re.findall(r'\d+\.\s*(?:Question|Q):\s*(.*?)\n\s*(?:Answer|A):\s*(.*?)(?=\n\d+\.\s*(?:Question|Q):|$)', raw, re.DOTALL | re.IGNORECASE)
    qa = [{"question": q.strip(), "answer": a.strip()} for q, a in matches2]
    return qa

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
    print(f"🚀 Doctor Run (Groq only) @ {datetime.now()}")
    book = process_uploaded_books()
    search_data = search_all_unrestricted()
    medical_data = query_all_medical_apis()
    hour = datetime.utcnow().hour
    serp = search_serpapi() if hour in [0,8,16] else ""
    combined = book + "\n" + search_data + "\n" + medical_data + "\n" + serp
    print(f"📊 Total chars: {len(combined)}")
    
    raw = ask_groq(combined)
    entries = parse_qa_text(raw)
    print(f"✅ Total entries extracted: {len(entries)}")
    
    if entries:
        out = get_output_file()
        with open(out, "a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"📁 Written to {out}")
    else:
        print("⚠️ No entries generated. Check preview above.")

if __name__ == "__main__":
    main()
