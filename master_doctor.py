import os, re, json, time
from datetime import datetime
from agents.search_team import search_all_unrestricted, search_serpapi
from agents.medical_team import query_all_medical_apis
import pypdf
from groq import Groq

def ask_groq(text):
    key = os.getenv("GROQ_API_KEY")
    if not key: return ""
    client = Groq(api_key=key)
    prompt = f"Generate 10 medical Q&A pairs. Use format:\nQuestion: <question>\nAnswer: <answer>\n\nText:\n{text[:4000]}"
    for attempt in range(3):
        try:
            chat = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Groq-এ নিশ্চিত, কাজ করবে
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return chat.choices[0].message.content
        except Exception as e:
            print(f"Groq err attempt {attempt+1}: {e}")
            time.sleep(10)
    return ""

def parse_qa_text(raw):
    qa = []
    matches = re.findall(r'(?:Question|Q):\s*(.*?)\n\s*(?:Answer|A):\s*(.*?)(?=\n\s*(?:Question|Q):|$)', raw, re.DOTALL | re.IGNORECASE)
    for q, a in matches:
        qa.append({"question": q.strip(), "answer": a.strip()})
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
    print(f"🚀 Doctor Run (Groq) @ {datetime.now()}")
    book = process_uploaded_books()
    search_data = search_all_unrestricted()
    medical_data = query_all_medical_apis()
    hour = datetime.utcnow().hour
    serp = search_serpapi() if hour in [0,8,16] else ""
    combined = book + "\n" + search_data + "\n" + medical_data + "\n" + serp
    print(f"📊 Chars: {len(combined)}")
    
    raw = ask_groq(combined)
    entries = parse_qa_text(raw)
    print(f"✅ {len(entries)} entries")
    
    if entries:
        out = get_output_file()
        with open(out, "a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"Written to {out}")

if __name__ == "__main__":
    main()
