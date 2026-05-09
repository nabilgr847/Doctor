import os, json, time, requests
from datetime import datetime
from agents.search_team import search_all_unrestricted, search_serpapi
from agents.medical_team import query_all_medical_apis
import pypdf

def call_huggingface(text):
    url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    prompt = f"Generate 15 unique medical Q&A pairs in JSON array format. Output ONLY the JSON array, no extra text.\nContext:\n{text[:4000]}"
    for i in range(3):
        try:
            r = requests.post(url, json={"inputs": prompt}, timeout=90)
            if r.status_code == 200:
                data = r.json()
                # Hugging Face আউটপুট কখনো dict, কখনো list-এ আসে
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'generated_text' in data:
                    return json.loads(data['generated_text'])
                else:
                    return json.loads(data)
        except:
            pass
        time.sleep(30)
    return []

def call_groq(text):
    key = os.getenv("GROQ_API_KEY")
    if not key: return []
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": "llama3-8b-8192", "messages": [{"role": "user", "content": f"Generate 15 unique medical Q&A pairs in JSON array format. Output ONLY the JSON array, no extra text.\nContext:\n{text[:4000]}"}], "temperature": 0.7}
    for i in range(3):
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=90)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                return json.loads(content)
        except:
            pass
        time.sleep(10)
    return []

def call_deepseek(text):
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key: return []
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": f"Generate 15 unique medical Q&A pairs in JSON array format. Output ONLY the JSON array, no extra text.\nContext:\n{text[:4000]}"}]}
    for i in range(3):
        try:
            r = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload, timeout=90)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                return json.loads(content)
        except:
            pass
        time.sleep(10)
    return []

def call_gemini(text):
    key = os.getenv("GEMINI_API_KEY")
    if not key: return []
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Generate 15 unique medical Q&A pairs in JSON array format. Output ONLY the JSON array, no extra text.\nContext:\n{text[:4000]}",
            config=types.GenerateContentConfig(temperature=0.7)
        )
        return json.loads(response.text)
    except:
        return []

def call_github_model(text, token):
    if not token: return []
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": f"Generate 15 unique medical Q&A pairs in JSON array format. Output ONLY the JSON array, no extra text.\nContext:\n{text[:4000]}"}], "temperature": 0.7}
    try:
        r = requests.post("https://models.inference.ai.azure.com/v1/chat/completions", headers=headers, json=payload, timeout=90)
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            return json.loads(content)
    except:
        pass
    return []

def process_uploaded_books():
    book_text = ""
    folder = "upload_books"
    if not os.path.exists(folder): return book_text
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder, filename)
            try:
                with open(filepath, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    pages_text = []
                    for page_num in range(min(10, len(reader.pages))):
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        if text: pages_text.append(text)
                    book_text += f"\n--- {filename} ---\n" + "\n".join(pages_text)
                processed_folder = "upload_books/processed"
                os.makedirs(processed_folder, exist_ok=True)
                os.rename(filepath, os.path.join(processed_folder, filename))
            except Exception as e:
                print(f"❌ {filename} সমস্যা: {e}")
    return book_text

def get_current_output_file():
    base, ext = "dataset", ".jsonl"
    counter = 1
    filename = f"{base}{ext}"
    while os.path.exists(filename):
        if os.path.getsize(filename) < 500 * 1024 * 1024:
            return filename
        counter += 1
        filename = f"{base}_{counter}{ext}"
    return filename

def main():
    print(f"🚀 Doctor Run @ {datetime.now()}")
    book_data = process_uploaded_books()
    search_data = search_all_unrestricted()
    medical_data = query_all_medical_apis()
    current_hour = datetime.utcnow().hour
    serp_data = search_serpapi() if current_hour in [0, 8, 16] else ""
    combined_text = book_data + "\n" + search_data + "\n" + medical_data + "\n" + serp_data
    print(f"📊 মোট ডেটা দৈর্ঘ্য: {len(combined_text)} অক্ষর")
    all_entries = []
    for func, name in [(call_huggingface, "Hugging Face"), (call_groq, "Groq"), (call_deepseek, "DeepSeek"), (call_gemini, "Gemini")]:
        print(f"🧠 {name} কাজ করছে...")
        entries = func(combined_text)
        if isinstance(entries, list):
            all_entries.extend(entries)
        print(f"  -> {len(entries) if isinstance(entries, list) else 0} entries")
    print("🧠 GitHub Models কাজ করছে...")
    token = os.getenv("GH_TOKEN")
    entries = call_github_model(combined_text, token)
    if isinstance(entries, list):
        all_entries.extend(entries)
    print(f"  -> {len(entries) if isinstance(entries, list) else 0} entries")
    if all_entries:
        output_file = get_current_output_file()
        with open(output_file, "a", encoding="utf-8") as f:
            for entry in all_entries:
                if isinstance(entry, dict) and "question" in entry:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"✅ {len(all_entries)} ডেটাসেট {output_file}-তে জমা হয়েছে।")
    else:
        print("⚠️ কোনো ডেটাসেট জেনারেট হয়নি।")

if __name__ == "__main__":
    main()
