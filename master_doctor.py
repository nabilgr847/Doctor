import os, re, json, time
from datetime import datetime, timedelta
from agents.search_team import search_all_unrestricted, search_serpapi
from agents.medical_team import query_all_medical_apis
import pypdf
from groq import Groq
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- Groq (এক Key দিয়ে চেষ্টা) ----------
def try_groq_with_key(key, text):
    client = Groq(api_key=key)
    models = ["openai/gpt-oss-120b", "llama-3.1-8b-instant"]
    for model in models:
        for _ in range(2):
            try:
                chat = client.chat.completions.create(
                    model=model,
                    messages=[{"role":"user","content":f"Generate exactly 50 concise medical Q&A pairs. Use format:\nQuestion: ...\nAnswer: ...\n\nText:\n{text[:2500]}"}],
                    temperature=0.7, max_tokens=4096)
                return chat.choices[0].message.content
            except Exception as e:
                print(f"❌ Groq error: {e}")
                time.sleep(5)
    return ""

# ---------- Pollinations ----------
def ask_pollinations_account(text):
    key = os.getenv("POLLINATIONS_API_KEY")
    if not key: return ""
    headers = {"Authorization": f"Bearer {key}", "Content-Type":"application/json"}
    data = {"messages":[{"role":"user","content":f"Generate exactly 50 concise medical Q&A pairs. Use format:\nQuestion: ...\nAnswer: ...\n\nText:\n{text[:2500]}"}],"model":"openai","temperature":0.7}
    try:
        r = requests.post("https://text.pollinations.ai/openai/v1/chat/completions", headers=headers, json=data, timeout=90)
        if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
    except: pass
    return ""

# ---------- OllamaFreeAPI ----------
def ask_ollamafree(text):
    try:
        from ollamafreeapi import OllamaFreeAPI
        return OllamaFreeAPI().chat(model="llama3.1:latest", prompt=f"Generate exactly 50 concise medical Q&A pairs. Use format:\nQuestion: ...\nAnswer: ...\n\nText:\n{text[:2500]}")
    except: return ""

# ---------- DeepSeek (লিমিট ফিরলে অটো চালু) ----------
def ask_deepseek(text):
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key: return ""
    headers = {"Authorization": f"Bearer {key}", "Content-Type":"application/json"}
    payload = {"model":"deepseek-chat","messages":[{"role":"user","content":f"Generate exactly 50 concise medical Q&A pairs. Use format:\nQuestion: ...\nAnswer: ...\n\nText:\n{text[:2500]}"}]}
    try:
        r = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload, timeout=90)
        if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
    except: pass
    return ""

# ---------- Gemini (লিমিট ফিরলে অটো চালু) ----------
def ask_gemini(text):
    key = os.getenv("GEMINI_API_KEY")
    if not key: return ""
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Generate exactly 50 concise medical Q&A pairs. Use format:\nQuestion: ...\nAnswer: ...\n\nText:\n{text[:2500]}")
        return response.text
    except: return ""

# ---------- Hugging Face (ফ্রি, ধীরে চলবে) ----------
def ask_huggingface(text):
    url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    for _ in range(2):
        try:
            r = requests.post(url, json={"inputs": f"Generate exactly 50 concise medical Q&A pairs. Use format:\nQuestion: ...\nAnswer: ...\n\nText:\n{text[:2500]}"}, timeout=90)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict) and 'generated_text' in data: return data['generated_text']
                elif isinstance(data, list) and len(data)>0: return data[0].get('generated_text', '')
        except: pass
        time.sleep(20)
    return ""

# ---------- উন্নত পার্সার ----------
def parse_qa_text(raw):
    if not raw: return []
    matches = re.findall(r'\d*\.?\s*(?:Question|Q):\s*(.*?)\n\s*(?:Answer|A):\s*(.*?)(?=\n\s*\d*\.?\s*(?:Question|Q):|$)', raw, re.DOTALL | re.IGNORECASE)
    qa = [{"question": q.strip(), "answer": a.strip()} for q, a in matches]
    if qa: return qa
    matches2 = re.findall(r'\*?\*?(?:Question|Q)\*?\*?:\s*(.*?)\n\s*\*?\*?(?:Answer|A)\*?\*?:\s*(.*?)(?=\n\s*\*?\*?(?:Question|Q)|$)', raw, re.DOTALL | re.IGNORECASE)
    return [{"question": q.strip(), "answer": a.strip()} for q, a in matches2]

# ---------- পিডিএফ ----------
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
            except Exception as e: print(f"PDF error {filename}: {e}")
    return book_text

def get_output_file():
    return f"dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"

def main():
    print(f"🚀 Doctor Non-Stop Run started @ {datetime.now()}")
    end_time = datetime.utcnow() + timedelta(hours=5, minutes=50)
    while datetime.utcnow() < end_time:
        start_cycle = time.time()
        book = process_uploaded_books()
        search_data = search_all_unrestricted()
        medical_data = query_all_medical_apis()
        hour = datetime.utcnow().hour
        serp = search_serpapi() if hour in [0,8,16] else ""
        combined = book + "\n" + search_data + "\n" + medical_data + "\n" + serp
        print(f"📊 Data length: {len(combined)}")
        all_raws = []
        # সর্বোচ্চ ৮টি সমান্তরাল কল
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            # Groq (2 keys)
            for key in [os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY_2")]:
                if key: futures.append(executor.submit(try_groq_with_key, key, combined))
            # Pollinations (2 calls)
            for _ in range(2): futures.append(executor.submit(ask_pollinations_account, combined))
            # OllamaFreeAPI (1)
            futures.append(executor.submit(ask_ollamafree, combined))
            # DeepSeek (1)
            futures.append(executor.submit(ask_deepseek, combined))
            # Gemini (1)
            futures.append(executor.submit(ask_gemini, combined))
            # Hugging Face (1)
            futures.append(executor.submit(ask_huggingface, combined))
            
            for future in as_completed(futures):
                res = future.result()
                if res: all_raws.append(res)
        
        entries = []
        for raw in all_raws: entries.extend(parse_qa_text(raw))
        print(f"📝 Total entries: {len(entries)}")
        if entries:
            out_file = get_output_file()
            with open(out_file, "w", encoding="utf-8") as f:
                for e in entries: f.write(json.dumps(e, ensure_ascii=False) + "\n")
            token = os.environ["GH_TOKEN"]
            repo = os.environ["REPOSITORY"]
            remote_url = f"https://x-access-token:{token}@github.com/{repo}.git"
            os.system("git config user.name 'God-Doctor-Bot'")
            os.system("git config user.email 'bot@doctor.ai'")
            os.system(f"git add {out_file}")
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
            os.system(f"git commit -m 'Dataset {ts}' || echo 'No changes'")
            os.system(f"git remote set-url origin {remote_url}")
            os.system("git push")
            print(f"✅ {len(entries)} entries pushed in {out_file}")
        else: print("⚠️ No entries this cycle.")
        elapsed = time.time() - start_cycle
        sleep_time = max(10, 15 - elapsed)
        print(f"⏳ Sleeping {sleep_time:.1f}s...")
        time.sleep(sleep_time)
    print("🏁 Non-stop run completed.")

if __name__ == "__main__":
    main()
