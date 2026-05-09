import os, re, json, time, threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pypdf
from groq import Groq

from agents.search_team import search_all_unrestricted
from agents.medical_team import query_all_medical_apis


# =========================
# 🔒 GLOBAL MEMORY
# =========================
seen_questions = set()
lock = threading.Lock()


# =========================
# 🧠 HELPERS
# =========================
def hash_q(q):
    return q.lower().strip()


def split_text(text, size=2000):
    return [text[i:i+size] for i in range(0, len(text), size)]


# =========================
# ⚛️ NUCLEUS MEDICAL PROMPT
# =========================
PROMPT = """
You are an advanced medical research intelligence system (PhD-level biomedical scientist + drug discovery AI).

TASK:
Generate EXACTLY 50 high-level medical research entries.

Each entry must be deep research-level (not simple Q&A).

========================
STRICT OUTPUT FORMAT (JSON ONLY):
========================
[
  {
    "question": "...",
    "answer": "...",
    "mechanism": "...",
    "drug_insight": "...",
    "future_innovation": "..."
  }
]

========================
REQUIREMENTS:
========================
- Identify disease nucleus (root cause)
- Explain molecular + cellular mechanism step-by-step
- Explain drug mechanism in detail
- Explain why current treatments fail (resistance, mutation, bypass)
- Suggest future biomedical innovation or technology

========================
QUESTION STYLE:
========================
- Why does disease X develop at molecular level?
- How does pathway Y control disease progression?
- Why do drugs fail in condition Z?
- What new therapy could solve this disease?

========================
RULES:
========================
- EXACTLY 50 entries
- No repetition
- No generic textbook answers
- No extra text outside JSON
- Focus: oncology, neurology, immunology, pharmacology, molecular biology

TEXT:
{text}
"""


# =========================
# 🤖 GROQ
# =========================
def groq_worker(key, text):
    try:
        client = Groq(api_key=key)

        chat = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": PROMPT.format(text=text[:5000])}],
            temperature=0.7,
            max_tokens=6000
        )

        return chat.choices[0].message.content
    except:
        return ""


# =========================
# 🌐 APIs
# =========================
def pollinations(text):
    try:
        r = requests.post(
            "https://text.pollinations.ai/openai/v1/chat/completions",
            json={"messages":[{"role":"user","content":text[:2000]}], "model":"openai"},
            timeout=60
        )
        return r.json()["choices"][0]["message"]["content"]
    except:
        return ""


def deepseek(text):
    try:
        r = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            json={
                "model":"deepseek-chat",
                "messages":[{"role":"user","content":text[:2000]}]
            },
            headers={"Authorization":f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"},
            timeout=60
        )
        return r.json()["choices"][0]["message"]["content"]
    except:
        return ""


def gemini(text):
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(text[:2000]).text
    except:
        return ""


def hf(text):
    try:
        r = requests.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-large",
            json={"inputs":text[:1500]},
            timeout=90
        )
        return r.json()[0]["generated_text"]
    except:
        return ""


def ollama(text):
    try:
        from ollamafreeapi import OllamaFreeAPI
        return OllamaFreeAPI().chat(
            model="llama3.1:latest",
            prompt=text[:2000]
        )
    except:
        return ""


# =========================
# 📄 PDF
# =========================
def pdf_text():
    folder = "upload_books"
    text = ""

    if not os.path.exists(folder):
        return ""

    for f in os.listdir(folder):
        if f.endswith(".pdf"):
            try:
                r = pypdf.PdfReader(os.path.join(folder, f))
                for p in r.pages[:3]:
                    t = p.extract_text()
                    if t:
                        text += t
            except:
                pass

    return text


# =========================
# 🧠 PARSER (JSON SAFE)
# =========================
def parse(raw):
    if not raw:
        return []

    try:
        data = json.loads(raw)
    except:
        return []

    out = []

    for item in data:
        q = item.get("question","").strip()
        a = item.get("answer","").strip()

        if not q or not a:
            continue

        key = hash_q(q)

        with lock:
            if key in seen_questions:
                continue
            seen_questions.add(key)

        out.append({
            "question": q,
            "answer": a,
            "mechanism": item.get("mechanism",""),
            "drug_insight": item.get("drug_insight",""),
            "future_innovation": item.get("future_innovation","")
        })

    return out


# =========================
# 🚀 MAIN ENGINE
# =========================
def main():
    print("🚀 NUCLEUS MEDICAL RESEARCH ENGINE STARTED")

    while True:

        start = time.time()

        base = (pdf_text() + search_all_unrestricted() + query_all_medical_apis())

        results = []

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = []

            for key in [os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY_2")]:
                if key:
                    futures.append(ex.submit(groq_worker, key, base))

            futures.append(ex.submit(pollinations, base))
            futures.append(ex.submit(deepseek, base))
            futures.append(ex.submit(gemini, base))
            futures.append(ex.submit(hf, base))
            futures.append(ex.submit(ollama, base))

            for f in as_completed(futures):
                r = f.result()
                if r:
                    results.append(r)

        # =========================
        # MERGE
        # =========================
        final = []
        for r in results:
            final.extend(parse(r))

        print("📝 OUTPUT:", len(final))

        # =========================
        # SAVE
        # =========================
        if final:
            file = f"dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"

            with open(file, "w", encoding="utf-8") as f:
                for i in final:
                    f.write(json.dumps(i, ensure_ascii=False) + "\n")

            print("✅ SAVED:", file)

        # =========================
        # SPEED CYCLE
        # =========================
        time.sleep(1.5)


if __name__ == "__main__":
    main()
