import os, re, json, time
from datetime import datetime, timedelta
from agents.search_team import search_all_unrestricted, search_serpapi
from agents.medical_team import query_all_medical_apis
import pypdf
from groq import Groq
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================== SYSTEM MESSAGE (গভীর উত্তর) ========================
SYSTEM_MESSAGE = (
    "You are a PhD‑level medical researcher writing for a medical textbook. "
    "Generate question‑answer pairs that are extremely detailed, explanatory, and 150–300 words each. "
    "Answers MUST include: (1) molecular/cellular mechanisms, (2) specific genes/proteins/pathways, "
    "(3) clinical relevance, (4) latest research or trials, (5) a concise summary. "
    "Answers must NEVER be one‑liners or simple definitions. "
    "Questions must be deep, analytical, and varied – rotate between What, How, Why, Compare, Discuss, "
    "Evaluate, Investigate, Elucidate, Explain the mechanism, and What is the role of [X] in [Y]. "
    "Every question‑answer pair MUST be UNIQUE."
)

# ======================== USER PROMPT (টপিক + ফরম্যাট) ========================
TOPICS = (
    "Molecular Biology, Genetics, Cancer, Immunology, Pharmacology, Cardiology, Neurology, Psychiatry, "
    "Infectious Diseases, Endocrinology, Gastroenterology, Nephrology, Pulmonology, Hematology, "
    "Dermatology, Ophthalmology, ENT, Orthopedics, Rheumatology, Pediatrics, Obstetrics, Radiology, "
    "Anesthesiology, Emergency Medicine, Public Health, Clinical Trials, Regenerative Medicine, "
    "Bioinformatics, Nanomedicine, Medical Ethics"
)

def make_user_prompt(count, text):
    return (
        f"Generate exactly {count} unique medical Q&A pairs. Topics: {TOPICS}. Rotate topics.\n"
        f"Format:\nQuestion: ...\nAnswer: ...\n\nText: {str(text)[:2500]}"
    )

# ======================== API ট্র্যাকিং ========================
api_tracker = {}

def update_tracker(name, ok, count=0, err=""):
    if name not in api_tracker:
        api_tracker[name] = {"status": "unknown", "last_error": "", "total": 0}
    t = api_tracker[name]
    if ok:
        t["status"] = "working"
        t["last_error"] = ""
        t["total"] += count
    else:
        t["status"] = "failed"
        t["last_error"] = err[:200] if err else "Unknown"

# ---------- Groq (2 keys, শুধু একটিভ মডেল) ----------
def try_groq_with_key(key, text, count=30, label="Groq"):
    client = Groq(api_key=key)
    # Groq-তে বর্তমানে সক্রিয় ও নির্ভরযোগ্য মডেলগুলোর তালিকা
    models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-4-scout-17b-16e-instruct",
        "qwen-3-32b",
        "deepseek-r1-distill-qwen-32b",
        "llama-3.2-90b-vision-preview"
    ]
    user_prompt = make_user_prompt(count, text)
    for model in models:
        print(f"🔄 {label} trying `{model}`...")
        for _ in range(2):
            try:
                chat = client.chat.completions.create(
                    model=model,
                    messages=[{"role":"system","content":SYSTEM_MESSAGE},{"role":"user","content":user_prompt}],
                    temperature=0.9, max_tokens=8192)
                print(f"✅ {label} success with `{model}`")
                return chat.choices[0].message.content
            except Exception as e:
                err = str(e)
                if "413" in err:
                    print(f"⏳ {label} `{model}` TPM limit, trying next...")
                else:
                    print(f"❌ {label} `{model}` error: {err[:150]}")
                    update_tracker(label, False, err=err)
                time.sleep(3)
    print(f"❌ {label} all models failed.")
    return ""

# ---------- Pollinations (পার্সিং ইম্প্রুভড) ----------
def ask_pollinations(text, count=30):
    key = os.getenv("POLLINATIONS_API_KEY")
    if not key:
        print("⚠️ Pollinations: no API key")
        return ""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    user_prompt = make_user_prompt(count, text)
    data = {"messages":[{"role":"system","content":SYSTEM_MESSAGE},{"role":"user","content":user_prompt}],"model":"openai","temperature":0.9}
    try:
        r = requests.post("https://text.pollinations.ai/openai/v1/chat/completions", headers=headers, json=data, timeout=90)
        if r.status_code == 200:
            resp = r.json()
            if "choices" in resp and len(resp["choices"]) > 0:
                return resp["choices"][0].get("message", {}).get("content", "")
            elif "content" in resp:
                return resp["content"]
            else:
                print(f"⚠️ Pollinations unexpected format: {str(resp)[:150]}")
                return ""
        else:
            print(f"❌ Pollinations HTTP {r.status_code}")
            update_tracker("Pollinations", False, err=f"HTTP {r.status_code}")
    except Exception as e:
        print(f"❌ Pollinations exception: {e}")
        update_tracker("Pollinations", False, err=str(e))
    return ""

# ---------- Mistral ----------
def ask_mistral(text, count=30):
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        print("⚠️ Mistral: no API key")
        return ""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    user_prompt = make_user_prompt(count, text)
    data = {"model":"mistral-small","messages":[{"role":"system","content":SYSTEM_MESSAGE},{"role":"user","content":user_prompt}],"temperature":0.9,"max_tokens":8192}
    try:
        r = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data, timeout=90)
        if r.status_code == 200:
            print("✅ Mistral success")
            return r.json()["choices"][0]["message"]["content"]
        else:
            print(f"❌ Mistral HTTP {r.status_code}")
            update_tracker("Mistral", False, err=f"HTTP {r.status_code}")
    except Exception as e:
        print(f"❌ Mistral exception: {e}")
        update_tracker("Mistral", False, err=str(e))
    return ""

# ---------- DevToolBox (ব্যাকআপ) ----------
def ask_devtoolbox(text, count=30):
    try:
        prompt = make_user_prompt(count, text)
        r = requests.post("https://devtoolbox-api.devtoolbox-api.workers.dev/ai/generate", json={"prompt": prompt}, timeout=90)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                return data.get("response", "")
    except Exception as e:
        update_tracker("DevToolBox", False, err=str(e))
    return ""

# ---------- পার্সার ----------
def parse_qa_text(raw, source="unknown"):
    if not raw: return []
    matches = re.findall(r'\d*\.?\s*(?:Question|Q):\s*(.*?)\n\s*(?:Answer|A):\s*(.*?)(?=\n\s*\d*\.?\s*(?:Question|Q):|$)', raw, re.DOTALL | re.IGNORECASE)
    qa = [{"question":q.strip(),"answer":a.strip(),"source":source} for q,a in matches]
    if qa: return qa
    matches2 = re.findall(r'\*?\*?(?:Question|Q)\*?\*?:\s*(.*?)\n\s*\*?\*?(?:Answer|A)\*?\*?:\s*(.*?)(?=\n\s*\*?\*?(?:Question|Q)|$)', raw, re.DOTALL | re.IGNORECASE)
    return [{"question":q.strip(),"answer":a.strip(),"source":source} for q,a in matches2]

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

# ---------- মেইন ----------
def main():
    print(f"🚀 Doctor Non-Stop Run started @ {datetime.now()}")
    end_time = datetime.utcnow() + timedelta(hours=5, minutes=50)
    qa_per_call = 30  # প্রতি API কলে 30 Q&A
    while datetime.utcnow() < end_time:
        start_cycle = time.time()
        book = process_uploaded_books()
        search_data = search_all_unrestricted()
        medical_data = query_all_medical_apis()
        hour = datetime.utcnow().hour
        serp = search_serpapi() if hour in [0, 8, 16] else ""
        combined = book + "\n" + search_data + "\n" + medical_data + "\n" + serp
        print(f"📊 Data length: {len(combined)}")

        all_raws = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            groq_keys = [
                ("Groq-1", os.getenv("GROQ_API_KEY")),
                ("Groq-2", os.getenv("GROQ_API_KEY_2"))
            ]
            for label, key in groq_keys:
                if key:
                    futures.append(executor.submit(
                        lambda l=label, k=key: (l, try_groq_with_key(k, combined, qa_per_call, l))))
            for _ in range(2):
                futures.append(executor.submit(
                    lambda: ("Pollinations", ask_pollinations(combined, qa_per_call))))
            futures.append(executor.submit(
                lambda: ("Mistral", ask_mistral(combined, qa_per_call))))
            futures.append(executor.submit(
                lambda: ("DevToolBox", ask_devtoolbox(combined, qa_per_call))))

            for future in as_completed(futures):
                source_name, raw = future.result()
                if raw:
                    all_raws.append((source_name, raw))

        entries = []
        entries_per_source = {}
        for source_name, raw in all_raws:
            parsed = parse_qa_text(raw, source=source_name)
            entries.extend(parsed)
            entries_per_source[source_name] = entries_per_source.get(source_name, 0) + len(parsed)
            update_tracker(source_name, True, count=len(parsed))

        print(f"📝 Total entries: {len(entries)}")
        print(f"📊 Sources: {entries_per_source}")
        print("📋 API Status Report:")
        for api_name, info in api_tracker.items():
            if info["status"] == "working":
                print(f"  ✅ {api_name}: working (total entries: {info['total']})")
            else:
                print(f"  ❌ {api_name}: failed - {info['last_error']}")

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
