import os, re, json, time
from datetime import datetime, timedelta
from agents.search_team import search_all_unrestricted, search_serpapi
from agents.medical_team import query_all_medical_apis
import pypdf
from groq import Groq
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================== সংক্ষিপ্ত প্রম্পট (Groq TPM <= 8000) ========================
PROMPT_TEMPLATE = """Generate exactly {count} UNIQUE medical question-answer pairs covering diverse medical fields.
Rotate topics each call (e.g., molecular biology, cancer, immunology, pharmacology, cardiology, neurology, infectious diseases, genetics, radiology, psychiatry, etc.).

ANSWERS must be:
- Extremely detailed, explanatory, PhD-level, 150-300 words
- Include mechanisms, genes/proteins, clinical relevance, research findings
- Never one-liners

QUESTIONS must be:
- Unique and varied (What, How, Why, Compare, Describe, Explain mechanism)
- Rotate topics each call

Format EXACTLY:
Question: ...
Answer: ...

Text: {text}"""

# ---------- Groq (এক Key, count=25) ----------
def try_groq_with_key(key, text, count=25):
    client = Groq(api_key=key)
    models = ["openai/gpt-oss-120b", "llama-3.1-8b-instant"]
    prompt = PROMPT_TEMPLATE.format(count=count, text=text[:2500])
    for model in models:
        for _ in range(2):
            try:
                chat = client.chat.completions.create(
                    model=model,
                    messages=[{"role":"user","content": prompt}],
                    temperature=0.9,
                    max_tokens=6144)   # বাড়ানো হয়েছে ২৫ জোড়ার জন্য
                return chat.choices[0].message.content
            except Exception as e:
                print(f"❌ Groq error: {e}")
                time.sleep(5)
    return ""

# ---------- Pollinations ----------
def ask_pollinations_account(text, count=25):
    key = os.getenv("POLLINATIONS_API_KEY")
    if not key: return ""
    headers = {"Authorization": f"Bearer {key}", "Content-Type":"application/json"}
    prompt = PROMPT_TEMPLATE.format(count=count, text=text[:2500])
    data = {"messages":[{"role":"user","content": prompt}],"model":"openai","temperature":0.9}
    try:
        r = requests.post("https://text.pollinations.ai/openai/v1/chat/completions", headers=headers, json=data, timeout=90)
        if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
    except: pass
    return ""

# ---------- OllamaFreeAPI ----------
def ask_ollamafree(text, count=25):
    prompt = PROMPT_TEMPLATE.format(count=count, text=text[:2500])
    try:
        from ollamafreeapi import OllamaFreeAPI
        return OllamaFreeAPI().chat(model="llama3.1:latest", prompt=prompt)
    except: return ""

# ---------- DeepSeek ----------
def ask_deepseek(text, count=25):
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key: return ""
    headers = {"Authorization": f"Bearer {key}", "Content-Type":"application/json"}
    prompt = PROMPT_TEMPLATE.format(count=count, text=text[:2500])
    payload = {"model":"deepseek-chat","messages":[{"role":"user","content": prompt}]}
    try:
        r = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload, timeout=90)
        if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
    except: pass
    return ""

# ---------- Gemini ----------
def ask_gemini(text, count=25):
    key = os.getenv("GEMINI_API_KEY")
    if not key: return ""
    prompt = PROMPT_TEMPLATE.format(count=count, text=text[:2500])
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except: return ""

# ---------- Hugging Face ----------
def ask_huggingface(text, count=25):
    url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    prompt = PROMPT_TEMPLATE.format(count=count, text=text[:2500])
    for _ in range(2):
        try:
            r = requests.post(url, json={"inputs": prompt}, timeout=90)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict) and 'generated_text' in data: return data['generated_text']
        except: pass
        time.sleep(20)
    return ""

# ---------- উন্নত পার্সার (source সহ) ----------
def parse_qa_text(raw, source="unknown"):
    if not raw: return []
    matches = re.findall(r'\d*\.?\s*(?:Question|Q):\s*(.*?)\n\s*(?:Answer|A):\s*(.*?)(?=\n\s*\d*\.?\s*(?:Question|Q):|$)', raw, re.DOTALL | re.IGNORECASE)
    qa = [{"question": q.strip(), "answer": a.strip(), "source": source} for q, a in matches]
    if qa: return qa
    matches2 = re.findall(r'\*?\*?(?:Question|Q)\*?\*?:\s*(.*?)\n\s*\*?\*?(?:Answer|A)\*?\*?:\s*(.*?)(?=\n\s*\*?\*?(?:Question|Q)|$)', raw, re.DOTALL | re.IGNORECASE)
    return [{"question": q.strip(), "answer": a.strip(), "source": source} for q, a in matches2]

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
    qa_per_call = 25  # বাড়ানো হলো
    while datetime.utcnow() < end_time:
        start_cycle = time.time()
        book = process_uploaded_books()
        search_data = search_all_unrestricted()
        medical_data = query_all_medical_apis()
        hour = datetime.utcnow().hour
        serp = search_serpapi() if hour in [0,8,16] else ""
        combined = book + "\n" + search_data + "\n" + medical_data + "\n" + serp
        print(f"📊 Data length: {len(combined)}")

        # প্যারালাল API কল (source সহ টাপল)
        all_raws = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for key in [os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY_2")]:
                if key:
                    futures.append(executor.submit(lambda k=key: ("Groq", try_groq_with_key(k, combined, qa_per_call))))
            for _ in range(2):
                futures.append(executor.submit(lambda: ("Pollinations", ask_pollinations_account(combined, qa_per_call))))
            futures.append(executor.submit(lambda: ("OllamaFreeAPI", ask_ollamafree(combined, qa_per_call))))
            futures.append(executor.submit(lambda: ("DeepSeek", ask_deepseek(combined, qa_per_call))))
            futures.append(executor.submit(lambda: ("Gemini", ask_gemini(combined, qa_per_call))))
            futures.append(executor.submit(lambda: ("HuggingFace", ask_huggingface(combined, qa_per_call))))

            for future in as_completed(futures):
                source_name, raw = future.result()
                if raw:
                    all_raws.append((source_name, raw))

        # পার্সিং এবং প্রতি API-র এন্ট্রি গণনা
        entries = []
        entries_per_source = {}
        for source_name, raw in all_raws:
            parsed = parse_qa_text(raw, source=source_name)
            entries.extend(parsed)
            entries_per_source[source_name] = entries_per_source.get(source_name, 0) + len(parsed)

        print(f"📝 Total entries: {len(entries)}")
        print(f"📊 Sources: {entries_per_source}")   # কীভাবে ভাগ হয়েছে তা লগে দেখাবে

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
        else:
            print("⚠️ No entries this cycle.")

        elapsed = time.time() - start_cycle
        sleep_time = max(5, 15 - elapsed)
        print(f"⏳ Sleeping {sleep_time:.1f}s...")
        time.sleep(sleep_time)
    print("🏁 Non-stop run completed.")

if __name__ == "__main__":
    main()
