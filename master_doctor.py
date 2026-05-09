import os, re, json, time, requests
from datetime import datetime
from agents.search_team import search_all_unrestricted, search_serpapi
from agents.medical_team import query_all_medical_apis
import pypdf

# ---------- LLM কল (কাঁচা রেসপন্স লগ করবে) ----------
def ask_llm(api_name, text):
    prompt = f"Generate 10 unique medical question-answer pairs from the text below.\nUse exactly this format:\nQuestion: <question>\nAnswer: <answer>\n\nText:\n{text[:4000]}"  # প্রম্পট সীমিত ৪০০০
    
    raw_response = ""
    
    # Hugging Face
    if api_name == "huggingface":
        for _ in range(3):
            try:
                r = requests.post("https://api-inference.huggingface.co/models/google/flan-t5-large",
                                  json={"inputs": prompt}, timeout=90)
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list) and len(data) > 0:
                        raw_response = data[0].get("generated_text", "")
                    elif isinstance(data, dict):
                        raw_response = data.get("generated_text", "")
                    break
            except Exception as e:
                print(f"  [HF error: {e}]")
            time.sleep(30)
    
    # Groq
    elif api_name == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key: return ""
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": "llama3-8b-8192", "messages": [{"role":"user","content": prompt}], "temperature": 0.7}
        for _ in range(3):
            try:
                r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=90)
                if r.status_code == 200:
                    raw_response = r.json()["choices"][0]["message"]["content"]
                    break
                else:
                    print(f"  [Groq HTTP {r.status_code}]")
            except Exception as e:
                print(f"  [Groq error: {e}]")
            time.sleep(10)
    
    # DeepSeek
    elif api_name == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key: return ""
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": "deepseek-chat", "messages": [{"role":"user","content": prompt}]}
        for _ in range(3):
            try:
                r = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload, timeout=90)
                if r.status_code == 200:
                    raw_response = r.json()["choices"][0]["message"]["content"]
                    break
                else:
                    print(f"  [DeepSeek HTTP {r.status_code}]")
            except Exception as e:
                print(f"  [DeepSeek error: {e}]")
            time.sleep(10)
    
    # Gemini
    elif api_name == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key: return ""
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        try:
            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt, config=types.GenerateContentConfig(temperature=0.7))
            raw_response = response.text
        except Exception as e:
            print(f"  [Gemini error: {e}]")
    
    # GitHub Models
    elif api_name == "github":
        token = os.getenv("GH_TOKEN")
        if not token: return ""
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"model": "gpt-4o-mini", "messages": [{"role":"user","content": prompt}], "temperature": 0.7}
        try:
            r = requests.post("https://models.inference.ai.azure.com/v1/chat/completions", headers=headers, json=payload, timeout=90)
            if r.status_code == 200:
                raw_response = r.json()["choices"][0]["message"]["content"]
            else:
                print(f"  [GitHub Models HTTP {r.status_code}]")
        except Exception as e:
            print(f"  [GitHub Models error: {e}]")
    
    print(f"  [Raw from {api_name}] {raw_response[:300]}")  # ডিবাগ
    return raw_response

# ---------- ফ্লেক্সিবল Q&A পার্সিং ----------
def parse_qa_text(raw_text):
    if not raw_text: return []
    qa_pairs = []
    # 1. "Question: ... Answer: ..." অথবা "Q: ... A: ..." ধরবে
    matches = re.findall(r'(?:Question|Q):\s*(.*?)\n\s*(?:Answer|A):\s*(.*?)(?=\n\s*(?:Question|Q):|$)',
                         raw_text, re.DOTALL | re.IGNORECASE)
    if matches:
        for q, a in matches:
            qa_pairs.append({"question": q.strip(), "answer": a.strip()})
        return qa_pairs
    
    # 2. যদি JSON অ্যারে থাকে (ফলব্যাক)
    try:
        json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if isinstance(data, list):
                for item in data:
                    if "question" in item and "answer" in item:
                        qa_pairs.append({"question": item["question"], "answer": item["answer"]})
    except:
        pass
    return qa_pairs

# ---------- পিডিএফ প্রসেসিং ----------
def process_uploaded_books():
    book_text = ""
    folder = "upload_books"
    if not os.path.exists(folder): return book_text
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder, filename)
            try:
                reader = pypdf.PdfReader(filepath)
                pages_text = []
                for page in reader.pages[:10]:
                    text = page.extract_text()
                    if text: pages_text.append(text)
                book_text += f"\n--- {filename} ---\n" + "\n".join(pages_text)
                processed_dir = os.path.join(folder, "processed")
                os.makedirs(processed_dir, exist_ok=True)
                os.replace(filepath, os.path.join(processed_dir, filename))
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
    print(f"🚀 Doctor Run @ {datetime.now()}")
    book_data = process_uploaded_books()
    search_data = search_all_unrestricted()
    medical_data = query_all_medical_apis()
    hour = datetime.utcnow().hour
    serp_data = search_serpapi() if hour in [0,8,16] else ""
    combined_text = book_data + "\n" + search_data + "\n" + medical_data + "\n" + serp_data
    print(f"📊 Total chars: {len(combined_text)}")
    
    all_entries = []
    apis = ["huggingface", "groq", "deepseek", "gemini", "github"]
    for api in apis:
        print(f"🧠 {api} working...")
        raw = ask_llm(api, combined_text)
        entries = parse_qa_text(raw)
        all_entries.extend(entries)
        print(f"  -> {len(entries)} entries")
    
    if all_entries:
        out_file = get_output_file()
        with open(out_file, "a", encoding="utf-8") as f:
            for entry in all_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"✅ {len(all_entries)} entries written to {out_file}")
    else:
        print("⚠️ No entries generated.")

if __name__ == "__main__":
    main()
