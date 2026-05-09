import os, json, time, requests
from datetime import datetime
from agents.search_team import search_all_unrestricted, search_serpapi
from agents.medical_team import query_all_medical_apis

# ================== LLM ব্রেইন (সব API-ই স্বাধীন ও ফল্ট-টলারেন্ট) ==================

def call_huggingface(text):
    url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    prompt = f"Generate 15 unique medical Q&A pairs in JSON array format. Output ONLY the JSON array, no extra text.\nContext:\n{text[:4000]}"
    for i in range(3):
        try:
            r = requests.post(url, json={"inputs": prompt}, timeout=90)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        time.sleep(30)
    return []

def call_groq(text):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return []
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": f"Generate 15 unique medical Q&A pairs in JSON array format. Output ONLY the JSON array, no extra text.\nContext:\n{text[:4000]}"}],
        "temperature": 0.7
    }
    for i in range(3):
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=90)
            if r.status_code == 200:
                return json.loads(r.json()["choices"][0]["message"]["content"])
        except:
            pass
        time.sleep(10)
    return []

def call_deepseek(text):
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return []
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": f"Generate 15 unique medical Q&A pairs in JSON array format. Output ONLY the JSON array, no extra text.\nContext:\n{text[:4000]}"}]
    }
    for i in range(3):
        try:
            r = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload, timeout=90)
            if r.status_code == 200:
                return json.loads(r.json()["choices"][0]["message"]["content"])
        except:
            pass
        time.sleep(10)
    return []

def call_gemini(text):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return []
    import google.generativeai as genai
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(f"Generate 15 unique medical Q&A pairs in JSON array format. Output ONLY the JSON array, no extra text.\nContext:\n{text[:4000]}")
        return json.loads(response.text)
    except:
        return []

def call_github_model(text, token):
    if not token:
        return []
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": f"Generate 15 unique medical Q&A pairs in JSON array format. Output ONLY the JSON array, no extra text.\nContext:\n{text[:4000]}"}],
        "temperature": 0.7
    }
    try:
        r = requests.post("https://models.inference.ai.azure.com/v1/chat/completions", headers=headers, json=payload, timeout=90)
        if r.status_code == 200:
            return json.loads(r.json()["choices"][0]["message"]["content"])
    except:
        pass
    return []

# ================== ফাইল সাইজ ম্যানেজমেন্ট ==================

def get_current_output_file():
    """সর্বশেষ dataset ফাইল খুঁজে বের করবে, সাইজ 500MB-র বেশি হলে নতুন ফাইল বানাবে"""
    base = "dataset"
    ext = ".jsonl"
    counter = 1
    filename = f"{base}{ext}"
    while os.path.exists(filename):
        if os.path.getsize(filename) < 500 * 1024 * 1024:  # 500MB
            return filename
        counter += 1
        filename = f"{base}_{counter}{ext}"
    return filename

# ================== মূল কাজ ==================

def main():
    print(f"🚀 Doctor Run @ {datetime.now()}")
    
    # 1. তথ্য সংগ্রহ (সব এজেন্ট থেকে)
    print("🔍 সার্চ এজেন্ট কাজ করছে...")
    search_data = search_all_unrestricted()
    
    print("⚕️ মেডিকেল এজেন্ট কাজ করছে...")
    medical_data = query_all_medical_apis()
    
    # SerpAPI শুধু দিনে ৩ বার (লিমিট বাঁচাতে)
    current_hour = datetime.utcnow().hour
    serp_data = search_serpapi() if current_hour in [0, 8, 16] else ""
    if serp_data:
        print("🕒 SerpAPI কল করা হয়েছে")
    
    # সব তথ্য একত্রিত
    combined_text = search_data + "\n" + medical_data + "\n" + serp_data
    print(f"📊 মোট ডেটা দৈর্ঘ্য: {len(combined_text)} অক্ষর")
    
    # 2. LLM ব্রেইন দিয়ে Q&A জেনারেশন
    all_entries = []
    
    # Hugging Face
    print("🧠 Hugging Face কাজ করছে...")
    entries = call_huggingface(combined_text)
    if isinstance(entries, list):
        all_entries.extend(entries)
    print(f"  -> {len(entries)} entries")
    
    # Groq
    print("🧠 Groq কাজ করছে...")
    entries = call_groq(combined_text)
    if isinstance(entries, list):
        all_entries.extend(entries)
    print(f"  -> {len(entries)} entries")
    
    # DeepSeek
    print("🧠 DeepSeek কাজ করছে...")
    entries = call_deepseek(combined_text)
    if isinstance(entries, list):
        all_entries.extend(entries)
    print(f"  -> {len(entries)} entries")
    
    # Gemini
    print("🧠 Gemini কাজ করছে...")
    entries = call_gemini(combined_text)
    if isinstance(entries, list):
        all_entries.extend(entries)
    print(f"  -> {len(entries)} entries")
    
    # GitHub Models (প্রথম টোকেন)
    print("🧠 GitHub Models কাজ করছে...")
    token = os.getenv("GH_TOKEN")
    entries = call_github_model(combined_text, token)
    if isinstance(entries, list):
        all_entries.extend(entries)
    print(f"  -> {len(entries)} entries")
    
    # 3. ফাইলে লেখা
    output_file = get_current_output_file()
    with open(output_file, "a", encoding="utf-8") as f:
        for entry in all_entries:
            if isinstance(entry, dict) and "question" in entry:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"✅ {len(all_entries)} টি নতুন ডেটাসেট {output_file}-তে জমা হয়েছে। ({datetime.now()})")

if __name__ == "__main__":
    main()
