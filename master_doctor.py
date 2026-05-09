import os, re, json, time, random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pypdf
from groq import Groq

from agents.search_team import search_all_unrestricted, search_serpapi
from agents.medical_team import query_all_medical_apis

# =========================
# 🔧 DEDUP MEMORY
# =========================
seen_questions = set()

# =========================
# 🔥 SAFE GROQ CALL
# =========================
def try_groq_with_key(key, text):
    client = Groq(api_key=key)

    models = [
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b"
    ]

    prompt = f"""
Generate 50 HIGH-QUALITY medical Q&A pairs.

RULES:
- No repeated questions
- No generic definitions
- Focus on clinical mechanisms, biomarkers, drug resistance, pathways
- Each question must be unique and medically meaningful

FORMAT:
Question: ...
Answer: ...

TEXT:
{text[:2000]}
"""

    for model in models:
        try:
            chat = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=3000
            )
            return chat.choices[0].message.content
        except Exception as e:
            print(f"❌ Groq error: {e}")
            time.sleep(3)

    return ""

# =========================
# 🧠 PARSE QA
# =========================
def parse_qa_text(raw):
    if not raw:
        return []

    matches = re.findall(
        r'(?:Question|Q):\s*(.*?)\n\s*(?:Answer|A):\s*(.*?)(?=\n(?:Question|Q):|$)',
        raw,
        re.DOTALL | re.IGNORECASE
    )

    result = []
    for q, a in matches:
        q = q.strip()
        a = a.strip()

        # ❌ DUPLICATE FILTER
        if q.lower() in seen_questions:
            continue

        seen_questions.add(q.lower())
        result.append({"question": q, "answer": a})

    return result

# =========================
# 📄 PDF PROCESS
# =========================
def process_uploaded_books():
    text = ""
    folder = "upload_books"

    if not os.path.exists(folder):
        return text

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            try:
                path = os.path.join(folder, file)
                reader = pypdf.PdfReader(path)

                pages = [p.extract_text() for p in reader.pages[:5] if p.extract_text()]
                text += "\n".join(pages)

                os.makedirs(f"{folder}/processed", exist_ok=True)
                os.replace(path, f"{folder}/processed/{file}")

            except Exception as e:
                print(f"PDF error: {e}")

    return text

# =========================
# 📁 OUTPUT FILE
# =========================
def get_output_file():
    return f"dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"

# =========================
# 🚀 MAIN LOOP
# =========================
def main():
    print("🚀 Doctor AI Pipeline started")

    end_time = datetime.utcnow() + timedelta(hours=5)

    while datetime.utcnow() < end_time:

        cycle_start = time.time()

        # -------- DATA SOURCES --------
        book = process_uploaded_books()
        search_data = search_all_unrestricted()
        medical_data = query_all_medical_apis()

        combined_sources = random.choice([
            book,
            search_data,
            medical_data
        ])

        print(f"📊 Input size: {len(combined_sources)}")

        # -------- PARALLEL AI CALLS --------
        all_raws = []

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []

            for key in [
                os.getenv("GROQ_API_KEY"),
                os.getenv("GROQ_API_KEY_2")
            ]:
                if key:
                    futures.append(executor.submit(try_groq_with_key, key, combined_sources))

            for future in as_completed(futures):
                res = future.result()
                if res:
                    all_raws.append(res)

        # -------- PARSE --------
        entries = []
        for raw in all_raws:
            entries.extend(parse_qa_text(raw))

        print(f"📝 Clean entries: {len(entries)}")

        # -------- SAVE --------
        if entries:
            file = get_output_file()

            with open(file, "w", encoding="utf-8") as f:
                for e in entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")

            # -------- GIT PUSH --------
            token = os.environ["GH_TOKEN"]
            repo = os.environ["REPOSITORY"]

            os.system("git config user.name 'God-Doctor-Bot'")
            os.system("git config user.email 'bot@doctor.ai'")

            os.system(f"git add {file}")
            os.system(f"git commit -m 'Dataset {datetime.utcnow()}' || echo 'No changes'")

            remote = f"https://x-access-token:{token}@github.com/{repo}.git"
            os.system(f"git remote set-url origin {remote}")
            os.system("git push")

            print(f"✅ Saved: {file}")

        else:
            print("⚠️ No entries generated")

        # =========================
        # ⏳ FIXED SLEEP TIME (5 sec)
        # =========================
        elapsed = time.time() - cycle_start
        sleep_time = max(5, 5 - elapsed)

        print(f"⏳ Sleeping {sleep_time:.1f}s...")
        time.sleep(sleep_time)

    print("🏁 Finished run")

if __name__ == "__main__":
    main()
