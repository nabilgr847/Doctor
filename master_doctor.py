import os, json, time, random
from datetime import datetime

# =========================
# 📊 STATUS TRACKER
# =========================
status = {
    "current_api": "none",
    "last_run": "",
    "total_generated": 0
}


# =========================
# 🔁 API FUNCTIONS (WRAP YOURS HERE)
# =========================
def groq_api(text):
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":text}],
            max_tokens=1500
        )
        return r.choices[0].message.content
    except:
        return ""


def deepseek_api(text):
    try:
        import requests
        r = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            json={"model":"deepseek-chat","messages":[{"role":"user","content":text}]},
            headers={"Authorization":f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"},
            timeout=20
        )
        return r.json()["choices"][0]["message"]["content"]
    except:
        return ""


def gemini_api(text):
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(text).text
    except:
        return ""


def pollinations_api(text):
    try:
        import requests
        r = requests.post(
            "https://text.pollinations.ai/openai/v1/chat/completions",
            json={"messages":[{"role":"user","content":text}]},
            timeout=20
        )
        return r.json()["choices"][0]["message"]["content"]
    except:
        return ""


# =========================
# 🔁 API ORDER (SEQUENTIAL)
# =========================
APIS = [
    ("GROQ", groq_api),
    ("DEEPSEEK", deepseek_api),
    ("GEMINI", gemini_api),
    ("POLLINATIONS", pollinations_api)
]


# =========================
# 🧠 PROMPT
# =========================
PROMPT = """
Generate 20 medical research entries.

Return JSON ONLY:
[
 {"question":"...","answer":"...","mechanism":"...","drug_insight":"...","future_innovation":"..."}
]
TEXT:
{t}
"""


# =========================
# 📦 SAFE PARSE
# =========================
def safe_json(raw):
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except:
        return []
    return []


# =========================
# 💾 SAVE
# =========================
def save(data):
    file = f"dataset_{int(time.time())}.jsonl"
    with open(file, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("💾 SAVED:", file)


# =========================
# 🔥 MAIN ENGINE (8 HOURS)
# =========================
def run_engine():

    start_time = time.time()
    EIGHT_HOURS = 8 * 60 * 60

    print("🚀 8-HOUR PIPELINE STARTED")

    while time.time() - start_time < EIGHT_HOURS:

        all_data = []

        prompt = PROMPT.format(t="medical nucleus research")

        # 🔁 SEQUENTIAL EXECUTION (IMPORTANT)
        for name, api in APIS:

            try:
                status["current_api"] = name
                print(f"⚡ Running: {name}")

                result = api(prompt)

                if result:
                    parsed = safe_json(result)
                    all_data.extend(parsed)

                time.sleep(random.uniform(2, 4))  # anti-rate-limit

            except:
                print(f"❌ Failed: {name}")
                continue

        # =========================
        # SAVE STEP
        # =========================
        if all_data:
            save(all_data)
            status["total_generated"] += len(all_data)

        status["last_run"] = datetime.utcnow().isoformat()

        print("📊 STATUS:", status)

        time.sleep(3)


# =========================
# 🚀 RUN
# =========================
if __name__ == "__main__":
    run_engine()
