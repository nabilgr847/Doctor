import os, json, time, threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pypdf
from groq import Groq

from agents.search_team import search_all_unrestricted
from agents.medical_team import query_all_medical_apis


# =========================
# 🔒 MEMORY (PERSISTENT)
# =========================
SEEN_FILE = "seen.json"
lock = threading.Lock()


def load_seen():
    if os.path.exists(SEEN_FILE):
        try:
            return set(json.load(open(SEEN_FILE)))
        except:
            return set()
    return set()


def save_seen(seen):
    try:
        json.dump(list(seen), open(SEEN_FILE, "w"))
    except:
        pass


seen_questions = load_seen()


# =========================
# 🧠 HELPERS
# =========================
def hash_q(q):
    return "".join(q.lower().split())


def split_text(text, size=1200):
    return [text[i:i+size] for i in range(0, len(text), size)]


# =========================
# ⚛️ PROMPT (ANTI-DUPLICATE)
# =========================
PROMPT = """
You are a PhD-level biomedical AI.

Generate EXACTLY 30 UNIQUE medical research entries in JSON ONLY.

RULES:
- No repetition
- Different diseases, pathways, mechanisms
- Deep molecular biology
- Drug mechanism + resistance
- Future innovation

FORMAT:
[
 {"question":"...","answer":"...","mechanism":"...","drug_insight":"...","future_innovation":"..."}
]

TEXT:
{text}
"""


# =========================
# 🔥 SAFE GROQ
# =========================
def groq_worker(key, text):
    try:
        if not key:
            return ""

        client = Groq(api_key=key)

        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": PROMPT.format(text=text[:1000])}],
            temperature=0.5,
            max_tokens=2500
        )

        return res.choices[0].message.content

    except:
        return ""


# =========================
# 🌐 SAFE REQUESTS
# =========================
def safe_post(url, payload, headers=None):
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=40)
        return r.json()
    except:
        return {}


def pollinations(text):
    j = safe_post(
        "https://text.pollinations.ai/openai/v1/chat/completions",
        {"messages":[{"role":"user","content":text[:1000]}]}
    )
    return j.get("choices",[{}])[0].get("message",{}).get("content","")


def deepseek(text):
    j = safe_post(
        "https://api.deepseek.com/v1/chat/completions",
        {"model":"deepseek-chat","messages":[{"role":"user","content":text[:1000]}]},
        {"Authorization":f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"}
    )
    return j.get("choices",[{}])[0].get("message",{}).get("content","")


def gemini(text):
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(text[:1000]).text
    except:
        return ""


def hf(text):
    j = safe_post(
        "https://api-inference.huggingface.co/models/google/flan-t5-large",
        {"inputs":text[:800]}
    )
    try:
        return j[0]["generated_text"]
    except:
        return ""


def ollama(text):
    try:
        from ollamafreeapi import OllamaFreeAPI
        return OllamaFreeAPI().chat(
            model="llama3.1:latest",
            prompt=text[:1000]
        )
    except:
        return ""


# =========================
# 📄 PDF
# =========================
def pdf_text():
    text = ""
    folder = "upload_books"

    if not os.path.exists(folder):
        return ""

    for f in os.listdir(folder):
        if f.endswith(".pdf"):
            try:
                r = pypdf.PdfReader(os.path.join(folder, f))
                for p in r.pages[:2]:
                    t = p.extract_text()
                    if t:
                        text += t
            except:
                pass

    return text[:2500]


# =========================
# 🧠 PARSER (SAFE + DEDUP)
# =========================
def parse(raw):
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

        out.append(item)

    return out


# =========================
# ⚡ API STACK
# =========================
def run_stack(base):

    tasks = [
        lambda: groq_worker(os.getenv("GROQ_API_KEY"), base),
        lambda: groq_worker(os.getenv("GROQ_API_KEY_2"), base),
        lambda: pollinations(base),
        lambda: deepseek(base),
        lambda: gemini(base),
        lambda: hf(base),
        lambda: ollama(base),
    ]

    results = []

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(t) for t in tasks]

        for f in as_completed(futures):
            try:
                r = f.result()
                if r:
                    results.append(r)
            except:
                pass

    return results


# =========================
# 🚀 MAIN ENGINE (STABLE LOOP)
# =========================
def main():
    print("🚀 STABLE NUCLEUS ENGINE STARTED")

    cycle = 0

    while cycle < 999999:

        base = (pdf_text() + search_all_unrestricted() + query_all_medical_apis())

        chunks = split_text(base)

        all_data = []

        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(run_stack, c) for c in chunks[:2]]

            for f in as_completed(futures):
                try:
                    all_data.extend(f.result())
                except:
                    pass

        final = []
        for r in all_data:
            final.extend(parse(r))

        print("📝 GENERATED:", len(final))

        if final:
            file = f"dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"

            with open(file, "w", encoding="utf-8") as f:
                for i in final:
                    f.write(json.dumps(i, ensure_ascii=False) + "\n")

            print("✅ SAVED:", file)

        save_seen(seen_questions)

        cycle += 1
        time.sleep(2)


if __name__ == "__main__":
    main()
