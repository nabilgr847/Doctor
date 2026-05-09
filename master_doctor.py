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
                    messages=[{
                        "role": "user",
                        "content": f"Generate exactly 50 concise medical Q&A pairs. Use format:\nQuestion: ...\nAnswer: ...\n\nText:\n{text[:2500]}"
                    }],
                    temperature=0.7,
                    max_tokens=4096
                )
                raw = chat.choices[0].message.content
                print(f"✅ Groq (key ending ...{key[-4:]}) success with {model}")
                return raw
            except Exception as e:
                print(f"❌ Groq key ending ...{key[-4:]} model {model} error: {e}")
                time.sleep(5)
    return ""

# ---------- Groq (দুটি Key প্যারালাল) ----------
def ask_groq_parallel(text):
    keys = [os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY_2")]
    valid_keys = [k for k in keys if k]
    results = []
    with ThreadPoolExecutor(max_workers=len(valid_keys)) as executor:
        futures = {executor.submit(try_groq_with_key, key, text): key for key in valid_keys}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
    return results

# ---------- Pollinations (অ্যাকাউন্ট Key ব্যবহার করে, 2টি প্যারালাল কল) ----------
def ask_pollinations_account(text):
    key = os.getenv("POLLINATIONS_API_KEY")
    if not key:
        return ""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data = {
        "messages": [{"role": "user", "content": f"Generate exactly 50 concise medical Q&A pairs. Use format:\nQuestion: ...\nAnswer: ...\n\nText:\n{text[:2500]}"}],
        "model": "openai",
        "temperature": 0.7
    }
    try:
        r = requests.post("https://text.pollinations.ai/openai/v1/chat/completions", headers=headers, json=data, timeout=90)
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            print(f"✅ Pollinations (account) success")
            return content
        else:
            print(f"❌ Pollinations error {r.status_code}: {r.text[:100]}")
    except Exception as e:
        print(f"❌ Pollinations exception: {e}")
    return ""

# ---------- Pollinations (2টি কল প্যারালাল) ----------
def ask_pollinations_parallel(text):
    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(ask_pollinations_account, text) for _ in range(2)]
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
    return results

# ---------- OllamaFreeAPI ----------
def ask_ollamafree(text):
    try:
        from ollamafreeapi import OllamaFreeAPI
        client = OllamaFreeAPI()
        res = client.chat(
            model="llama3.1:latest",
            prompt=f"Generate exactly 50 concise medical Q&A pairs. Use format:\nQuestion: ...\nAnswer: ...\n\nText:\n{text[:2500]}"
        )
        if res:
            print(f"✅ OllamaFreeAPI success")
            return res
    except Exception as e:
        print(f"❌ OllamaFreeAPI error: {e}")
    return ""

# ---------- উন্নত পার্সার ----------
def parse_qa_text(raw):
    if not raw:
        return []
    matches = re.findall(r'\d*\.?\s*(?:Question|Q):\s*(.*?)\n\s*(?:Answer|A):\s*(.*?)(?=\n\s*\d*\.?\s*(?:Question|Q):|$)', raw, re.DOTALL | re.IGNORECASE)
    qa = [{"question": q.strip(), "answer": a.strip()} for q, a in matches]
    if qa:
        return qa
    matches2 = re.findall(r'\*?\*?(?:Question|Q)\*?\*?:\s*(.*?)\n\s*\*?\*?(?:Answer|A)\*?\*?:\s*(.*?)(?=\n\s*\*?\*?(?:Question|Q)|$)', raw, re.DOTALL | re.IGNORECASE)
    return [{"question": q.strip(), "answer": a.strip()} for q, a in matches2]

# ---------- পিডিএফ ----------
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
        
        # তথ্য সংগ্রহ
        book = process_uploaded_books()
        search_data = search_all_unrestricted()
        medical_data = query_all_medical_apis()
        hour = datetime.utcnow().hour
        serp = search_serpapi() if hour in [0,8,16] else ""
        combined = book + "\n" + search_data + "\n" + medical_data + "\n" + serp
        print(f"📊 Data length: {len(combined)}")
        
        # সব API প্যারালাল
        all_raws = []
        # Groq (2 keys) parallel
        all_raws.extend(ask_groq_parallel(combined))
        # Pollinations (2 calls) parallel
        all_raws.extend(ask_pollinations_parallel(combined))
        # OllamaFreeAPI (1 call)
        olla = ask_ollamafree(combined)
        if olla:
            all_raws.append(olla)
        
        entries = []
        for raw in all_raws:
            entries.extend(parse_qa_text(raw))
        print(f"📝 Total entries: {len(entries)}")
        
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
            
            token = os.environ["GH_TOKEN"]
            repo = os.environ["REPOSITORY"]
            remote_url = f"https://x-access-token:{token}@github.com/{repo}.git"
            os.system(f"git remote set-url origin {remote_url}")
            os.system("git pull --rebase origin main")
            os.system("git push")
            
            print(f"✅ {len(entries)} entries pushed to repo")
        else:
            print("⚠️ No entries this cycle.")
        
        elapsed = time.time() - start_cycle
        sleep_time = max(10, 15 - elapsed)
        print(f"⏳ Sleeping {sleep_time:.1f}s...")
        time.sleep(sleep_time)
    
    print("🏁 Non-stop run completed. Next scheduled run will take over.")

if __name__ == "__main__":
    main()
