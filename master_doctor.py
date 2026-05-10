import os, re, json, time
from datetime import datetime, timedelta
from agents.search_team import search_all_unrestricted, search_serpapi
from agents.medical_team import query_all_medical_apis
import pypdf
from groq import Groq
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================== প্রম্পট (30 টপিক, 30 Q&A) ========================
PROMPT_TEMPLATE = """Generate exactly {count} UNIQUE medical question-answer pairs covering an extremely wide range of medical and biomedical fields.  
You MUST rotate topics each call so that no single field dominates. Use all of the following domains over time:  

1. Molecular & Cellular Biology  
2. Genetics & Epigenetics  
3. Cancer Biology & Oncology  
4. Immunology & Vaccinology  
5. Pharmacology & Toxicology  
6. Cardiology & Cardiovascular Surgery  
7. Neurology & Neurosurgery  
8. Psychiatry & Mental Health  
9. Infectious Diseases & Microbiology  
10. Endocrinology & Metabolism  
11. Gastroenterology & Hepatology  
12. Nephrology & Urology  
13. Pulmonology & Respiratory Medicine  
14. Hematology & Transfusion Medicine  
15. Dermatology & Skin Pathology  
16. Ophthalmology & Vision Science  
17. Otolaryngology (ENT)  
18. Orthopedics & Sports Medicine  
19. Rheumatology & Autoimmune Disorders  
20. Pediatrics & Neonatology  
21. Obstetrics & Gynecology  
22. Radiology & Medical Imaging  
23. Anesthesiology & Pain Management  
24. Emergency Medicine & Toxicology  
25. Public Health & Epidemiology  
26. Clinical Trials & Evidence-Based Medicine  
27. Regenerative Medicine & Stem Cells  
28. Bioinformatics & Computational Biology  
29. Nanomedicine & Drug Delivery  
30. Medical Ethics & Health Policy  

ANSWERS must be:  
- Extremely detailed, explanatory, PhD‑level, 150–300 words  
- Include mechanisms, genes/proteins, clinical relevance, research findings  
- Never one‑liners  

QUESTIONS must be:  
- Unique, varied (What, How, Why, Compare, Describe, Explain mechanism)  
- Rotate topics thoroughly  

Format EXACTLY:  
Question: ...  
Answer: ...  

Text: {text}"""

# ======================== API ট্র্যাকিং ========================
api_tracker = {}

def update_tracker(source_name, success, entries_count=0, error_msg=None):
    if source_name not in api_tracker:
        api_tracker[source_name] = {"status": "unknown", "last_error": None, "total_entries": 0}
    tracker = api_tracker[source_name]
    if success:
        tracker["status"] = "working"
        tracker["last_error"] = None
        tracker["total_entries"] += entries_count
    else:
        tracker["status"] = "failed"
        tracker["last_error"] = error_msg[:200] if error_msg else "Unknown error"

# ---------- Groq (30 Q&A) ----------
def try_groq_with_key(key, text, count=30):
    if not key: return ""
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
                    max_tokens=8192)
                return chat.choices[0].message.content
            except Exception as e:
                update_tracker("Groq", False, error_msg=str(e))
                time.sleep(5)
    return ""

# ---------- Pollinations ----------
def ask_pollinations_account(text, count=30):
    key = os.getenv("POLLINATIONS_API_KEY")
    if not key: return ""
    headers = {"Authorization": f"Bearer {key}", "Content-Type":"application/json"}
    prompt = PROMPT_TEMPLATE.format(count=count, text=text[:2500])
    data = {"messages":[{"role":"user","content": prompt}],"model":"openai","temperature":0.9}
    try:
        r = requests.post("https://text.pollinations.ai/openai/v1/chat/completions", headers=headers, json=data, timeout=90)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            update_tracker("Pollinations", False, error_msg=f"HTTP {r.status_code}")
    except Exception as e:
        update_tracker("Pollinations", False, error_msg=str(e))
    return ""

# ---------- OllamaFreeAPI ----------
def ask_ollamafree(text, count=30):
    prompt = PROMPT_TEMPLATE.format(count=count, text=text[:2500])
    try:
        from ollamafreeapi import OllamaFreeAPI
        res = OllamaFreeAPI().chat(model="llama3.1:latest", prompt=prompt)
        if res:
            return res
        else:
            update_tracker("OllamaFreeAPI", False, error_msg="Empty response")
    except Exception as e:
        update_tracker("OllamaFreeAPI", False, error_msg=str(e))
    return ""

# ---------- DeepSeek ----------
def ask_deepseek(text, count=30):
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key: return ""
    headers = {"Authorization": f"Bearer {key}", "Content-Type":"application/json"}
    prompt = PROMPT_TEMPLATE.format(count=count, text=text[:2500])
    payload = {"model":"deepseek-chat","messages":[{"role":"user","content": prompt}]}
    try:
        r = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload, timeout=90)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            update_tracker("DeepSeek", False, error_msg=f"HTTP {r.status_code}")
    except Exception as e:
        update_tracker("DeepSeek", False, error_msg=str(e))
    return ""

# ---------- Gemini ----------
def ask_gemini(text, count=30):
    key = os.getenv("GEMINI_API_KEY")
    if not key: return ""
    prompt = PROMPT_TEMPLATE.format(count=count, text=text[:2500])
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        update_tracker("Gemini", False, error_msg=str(e))
    return ""

# ---------- Hugging Face ----------
def ask_huggingface(text, count=30):
    url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    prompt = PROMPT_TEMPLATE.format(count=count, text=text[:2500])
    for _ in range(2):
        try:
            r = requests.post(url, json={"inputs": prompt}, timeout=90)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict) and 'generated_text' in data:
                    return data['generated_text']
            else:
                update_tracker("HuggingFace", False, error_msg=f"HTTP {r.status_code}")
        except Exception as e:
            update_tracker("HuggingFace", False, error_msg=str(e))
        time.sleep(20)
    return ""

# ---------- উন্নত পার্সার (source সহ) ----------
def parse_qa_text(raw, source="unknown"):
    if not raw: return []
    # প্রধান: Question/Answer বা Q/A (ঐচ্ছিক নম্বর সহ)
    matches = re.findall(r'\d*\.?\s*(?:Question|Q):\s*(.*?)\n\s*(?:Answer|A):\s*(.*?)(?=\n\s*\d*\.?\s*(?:Question|Q):|$)', raw, re.DOTALL | re.IGNORECASE)
    qa = [{"question": q.strip(), "answer": a.strip(), "source": source} for q, a in matches]
    if qa: return qa
    # ব্যাকআপ: **Question:** বা **Answer:**
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

# ---------- মেইন (non-stop লুপ) ----------
def main():
    print(f"🚀 Doctor Non-Stop Run started @ {datetime.now()}")
    end_time = datetime.utcnow() + timedelta(hours=5, minutes=50)
    qa_per_call = 30  # 30 Q&A per API
    while datetime.utcnow() < end_time:
        start_cycle = time.time()
        book = process_uploaded_books()
        search_data = search_all_unrestricted()
        medical_data = query_all_medical_apis()
        hour = datetime.utcnow().hour
        serp = search_serpapi() if hour in [0,8,16] else ""
        combined = book + "\n" + search_data + "\n" + medical_data + "\n" + serp
        print(f"📊 Data length: {len(combined)}")

        all_raws = []  # (source_name, raw_text)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            # Groq (2 keys)
            for key in [os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY_2")]:
                if key:
                    futures.append(executor.submit(lambda k=key: ("Groq", try_groq_with_key(k, combined, qa_per_call))))
            # Pollinations (2 calls)
            for _ in range(2):
                futures.append(executor.submit(lambda: ("Pollinations", ask_pollinations_account(combined, qa_per_call))))
            # OllamaFreeAPI
            futures.append(executor.submit(lambda: ("OllamaFreeAPI", ask_ollamafree(combined, qa_per_call))))
            # DeepSeek
            futures.append(executor.submit(lambda: ("DeepSeek", ask_deepseek(combined, qa_per_call))))
            # Gemini
            futures.append(executor.submit(lambda: ("Gemini", ask_gemini(combined, qa_per_call))))
            # Hugging Face
            futures.append(executor.submit(lambda: ("HuggingFace", ask_huggingface(combined, qa_per_call))))

            for future in as_completed(futures):
                source_name, raw = future.result()
                if raw:
                    all_raws.append((source_name, raw))

        # পার্সিং ও প্রতি API-র এন্ট্রি গণনা
        entries = []
        entries_per_source = {}
        for source_name, raw in all_raws:
            parsed = parse_qa_text(raw, source=source_name)
            entries.extend(parsed)
            entries_per_source[source_name] = entries_per_source.get(source_name, 0) + len(parsed)
            update_tracker(source_name, True, entries_count=len(parsed))

        print(f"📝 Total entries: {len(entries)}")
        print(f"📊 Sources: {entries_per_source}")
        print("📋 API Status Report:")
        for api_name, info in api_tracker.items():
            if info["status"] == "working":
                print(f"  ✅ {api_name}: working (total entries: {info['total_entries']})")
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
