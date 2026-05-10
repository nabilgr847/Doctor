import os, re, json, time
from datetime import datetime, timedelta
from agents.search_team import search_all_unrestricted, search_serpapi
from agents.medical_team import query_all_medical_apis
import pypdf
from groq import Groq
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================== SYSTEM MESSAGE (গভীরতা নিশ্চিত) ========================
SYSTEM_MESSAGE = (
    "You are a PhD‑level medical researcher writing for a medical textbook. "
    "Generate question‑answer pairs that are extremely detailed, explanatory, and 150–300 words each. "
    "Answers MUST include: (1) molecular/cellular mechanisms, (2) specific genes/proteins/pathways, "
    "(3) clinical relevance, (4) latest research or trials, (5) a concise summary. "
    "Answers must NEVER be one‑liners or simple definitions. "
    "Questions must be deep, analytical, and varied – rotate between What, How, Why, Compare, Discuss, "
    "Evaluate, Investigate, Elucidate, Explain the mechanism, and What is the role of [X] in [Y]. "
    "Every question‑answer pair MUST be UNIQUE and not repeat any previous pair in the dataset."
)

# ======================== USER PROMPT (টপিক + ফরম্যাট) ========================
# সরাসরি f‑string ব্যবহার করব, যাতে text ধরণের সমস্যা না হয়
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

# ---------- Groq (2 keys, আলাদা, নির্ভরযোগ্য মডেল) ----------
def try_groq_with_key(key, text, count=30, label="Groq"):
    client = Groq(api_key=key)
    models = [
        "openai/gpt-oss-120b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768"
    ]
    user_prompt = make_user_prompt(count, text)
    for model in models:
        for _ in range(2):
            try:
                chat = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.9,
                    max_tokens=8192
                )
                return chat.choices[0].message.content
            except Exception as e:
                err = str(e)
                if "413" not in err:
                    update_tracker(label, False, err=err)
                time.sleep(3)
    return ""

# ---------- Pollinations ----------
def ask_pollinations(text, count=30):
    key = os.getenv("POLLINATIONS_API_KEY")
    if not key: return ""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    user_prompt = make_user_prompt(count, text)
    data = {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_prompt}
        ],
        "model": "openai",
        "temperature": 0.9
    }
    try:
        r = requests.post("https://text.pollinations.ai/openai/v1/chat/completions",
                          headers=headers, json=data, timeout=90)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            update_tracker("Pollinations", False, err=f"HTTP {r.status_code}")
    except Exception as e:
        update_tracker("Pollinations", False, err=str(e))
    return ""

# ---------- Mistral ----------
def ask_mistral(text, count=30):
    key = os.getenv("MISTRAL_API_KEY")
    if not key: return ""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    user_prompt = make_user_prompt(count, text)
    data = {
        "model": "mistral-small",
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.9,
        "max_tokens": 8192
    }
    try:
        r = requests.post("https://api.mistral.ai/v1/chat/completions",
                          headers=headers, json=data, timeout=90)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            update_tracker("Mistral", False, err=f"HTTP {r.status_code}")
    except Exception as e:
        update_tracker("Mistral", False, err=str(e))
    return ""

# ---------- পার্সার ----------
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
        with ThreadPoolExecutor(max_workers=6) as executor:  # 2 Groq + 2 Pollinations + 1 Mistral = 5, একটু বেশি রাখা
            futures = []
            # Groq (2 keys)
            groq_keys = [
                ("Groq-1", os.getenv("GROQ_API_KEY")),
                ("Groq-2", os.getenv("GROQ_API_KEY_2"))
            ]
            for label, key in groq_keys:
                if key:
                    futures.append(executor.submit(
                        lambda l=label, k=key: (l, try_groq_with_key(k, combined, qa_per_call, l))))
            # Pollinations (2 calls)
            for _ in range(2):
                futures.append(executor.submit(
                    lambda: ("Pollinations", ask_pollinations(combined, qa_per_call))))
            # Mistral
            futures.append(executor.submit(
                lambda: ("Mistral", ask_mistral(combined, qa_per_call))))

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
