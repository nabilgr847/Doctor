import os, re, json, time
from datetime import datetime, timedelta
from agents.search_team import search_all_unrestricted, search_serpapi
from agents.medical_team import query_all_medical_apis
import pypdf
from groq import Groq
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================== মাস্টার প্রম্পট (গবেষণামূলক, ব্যাখ্যামূলক, মেডিকেলের সব শাখা) ========================
PROMPT_TEMPLATE = """Generate exactly {count} UNIQUE, research-grade medical question-answer pairs. 
You MUST cover DIFFERENT topics from the ENTIRE medical and biological sciences domain. 
Include at least all of these areas, rotating each time:
1. Molecular biology (DNA/RNA, gene editing, CRISPR, epigenetics)
2. Cancer biology (specific cancers: breast, lung, colorectal, pancreatic; mechanisms: metastasis, angiogenesis, tumor microenvironment)
3. Immunology (innate & adaptive immunity, checkpoint inhibitors, CAR-T, vaccines, autoimmune diseases)
4. Pharmacology (drug mechanisms, pharmacokinetics, adverse effects, drug-drug interactions)
5. Cardiology (atherosclerosis, heart failure, arrhythmias, hypertension)
6. Neurology (Alzheimer's, Parkinson's, stroke, neuroimaging, synapses)
7. Infectious diseases (antibiotics, antiviral, vaccines, emerging pathogens)
8. Endocrinology (diabetes, thyroid, hormonal regulation)
9. Genetics (inheritance patterns, GWAS, gene therapy)
10. Rare diseases (orphan drugs, case studies)
11. Diagnostic imaging (MRI, CT, PET, ultrasound principles)
12. Clinical trials (phases, blinding, endpoints, ethics)
13. Surgery (techniques, complications, perioperative care)
14. Pathology (histology, biomarkers, immunohistochemistry)
15. Bioinformatics (sequence alignment, protein structure prediction, omics data)
16. Epidemiology (study designs, bias, meta-analysis)
17. Public health (vaccination policies, outbreak investigation)
18. Regenerative medicine (stem cells, tissue engineering)
19. Nanomedicine (drug delivery, nanoparticles)
20. Psychiatry (depression, schizophrenia, psychotherapy)
21. Pediatrics (neonatology, growth disorders)
22. Dermatology (psoriasis, melanoma, skin infections)
23. Ophthalmology (retinal diseases, glaucoma)
24. Hematology (anemia, leukemia, transfusion)
25. Nephrology (kidney function, dialysis)
26. Gastroenterology (IBD, liver cirrhosis, microbiome)
27. Pulmonology (COPD, asthma, pneumonia)
28. Urology (prostate cancer, kidney stones)
29. Obstetrics & Gynecology (pregnancy, endometriosis)
30. Anesthesiology (types of anesthesia, pain management)
31. Emergency medicine (trauma, triage)
32. Radiology (X-ray, radiation therapy)
33. Medical ethics (informed consent, confidentiality)
34. Medical education (clinical reasoning, OSCE)
35. Healthcare systems (policy, insurance models)

CRITICAL INSTRUCTIONS FOR ANSWERS:
- Answers MUST be extremely detailed, comprehensive, and explanatory (like a PhD dissertation or a medical textbook).
- NEVER give one-liner answers. Each answer should develop the reasoning, mechanisms, clinical implications, and where relevant, mention research studies or guidelines.
- Structure complex answers logically: define key terms, explain the underlying biology/mechanism, discuss clinical relevance, mention any controversies or open questions, and end with a concise summary.
- For each answer, aim for at least 150-300 words. Go deeper if the topic demands it.
- Include references to specific genes, proteins, pathways, diagnostic criteria, staging systems, or clinical trial names when appropriate.
- You may use the collected text as a starting point, but you may enrich the answer with your internal medical knowledge.

Important rules for questions:
- Every question MUST BE UNIQUE and NOT repeat any previous question.
- Vary question starters: What, How, Why, Which, Compare, Describe, Discuss, Explain the mechanism, What is the role of, How does [X] affect [Y], etc.
- Questions should invite detailed, analytical answers.

Use format EXACTLY:
Question: ...
Answer: ...

Now, based on the following collected medical information, generate {count} pairs.
Text: {text}"""

# ---------- Groq (এক Key দিয়ে চেষ্টা) ----------
def try_groq_with_key(key, text, count=50):
    client = Groq(api_key=key)
    models = ["openai/gpt-oss-120b", "llama-3.1-8b-instant"]
    prompt = PROMPT_TEMPLATE.format(count=count, text=text[:2500])
    for model in models:
        for _ in range(2):
            try:
                chat = client.chat.completions.create(
                    model=model,
                    messages=[{"role":"user","content": prompt}],
                    temperature=0.9, max_tokens=8192 if count>=50 else 4096)
                return chat.choices[0].message.content
            except Exception as e:
                print(f"❌ Groq error: {e}")
                time.sleep(5)
    return ""

# ---------- Pollinations ----------
def ask_pollinations_account(text, count=50):
    key = os.getenv("POLLINATIONS_API_KEY")
    if not key: return ""
    headers = {"Authorization": f"Bearer {key}", "Content-Type":"application/json"}
    prompt = PROMPT_TEMPLATE.format(count=count, text=text[:2500])
    data = {"messages":[{"role":"user","content": prompt}],"model":"openai","temperature":0.9}
    try:
        r = requests.post("https://text.pollinations.ai/openai/v1/chat/completions", headers=headers, json=data, timeout=90)
        if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
    except: pass
    return ""

# ---------- OllamaFreeAPI ----------
def ask_ollamafree(text, count=50):
    prompt = PROMPT_TEMPLATE.format(count=count, text=text[:2500])
    try:
        from ollamafreeapi import OllamaFreeAPI
        return OllamaFreeAPI().chat(model="llama3.1:latest", prompt=prompt)
    except: return ""

# ---------- DeepSeek ----------
def ask_deepseek(text, count=50):
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key: return ""
    headers = {"Authorization": f"Bearer {key}", "Content-Type":"application/json"}
    prompt = PROMPT_TEMPLATE.format(count=count, text=text[:2500])
    payload = {"model":"deepseek-chat","messages":[{"role":"user","content": prompt}]}
    try:
        r = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload, timeout=90)
        if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
    except: pass
    return ""

# ---------- Gemini ----------
def ask_gemini(text, count=50):
    key = os.getenv("GEMINI_API_KEY")
    if not key: return ""
    prompt = PROMPT_TEMPLATE.format(count=count, text=text[:2500])
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except: return ""

# ---------- Hugging Face ----------
def ask_huggingface(text, count=50):
    url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    prompt = PROMPT_TEMPLATE.format(count=count, text=text[:2500])
    for _ in range(2):
        try:
            r = requests.post(url, json={"inputs": prompt}, timeout=90)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict) and 'generated_text' in data: return data['generated_text']
        except: pass
        time.sleep(20)
    return ""

# ---------- পার্সার ----------
def parse_qa_text(raw):
    if not raw: return []
    matches = re.findall(r'\d*\.?\s*(?:Question|Q):\s*(.*?)\n\s*(?:Answer|A):\s*(.*?)(?=\n\s*\d*\.?\s*(?:Question|Q):|$)', raw, re.DOTALL | re.IGNORECASE)
    qa = [{"question": q.strip(), "answer": a.strip()} for q, a in matches]
    if qa: return qa
    matches2 = re.findall(r'\*?\*?(?:Question|Q)\*?\*?:\s*(.*?)\n\s*\*?\*?(?:Answer|A)\*?\*?:\s*(.*?)(?=\n\s*\*?\*?(?:Question|Q)|$)', raw, re.DOTALL | re.IGNORECASE)
    return [{"question": q.strip(), "answer": a.strip()} for q, a in matches2]

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

def main():
    print(f"🚀 Doctor Non-Stop Run started @ {datetime.now()}")
    end_time = datetime.utcnow() + timedelta(hours=5, minutes=50)
    qa_per_call = 50
    while datetime.utcnow() < end_time:
        start_cycle = time.time()
        book = process_uploaded_books()
        search_data = search_all_unrestricted()
        medical_data = query_all_medical_apis()
        hour = datetime.utcnow().hour
        serp = search_serpapi() if hour in [0,8,16] else ""
        combined = book + "\n" + search_data + "\n" + medical_data + "\n" + serp
        print(f"📊 Data length: {len(combined)}")
        all_raws = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for key in [os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY_2")]:
                if key: futures.append(executor.submit(try_groq_with_key, key, combined, qa_per_call))
            for _ in range(2): futures.append(executor.submit(ask_pollinations_account, combined, qa_per_call))
            futures.append(executor.submit(ask_ollamafree, combined, qa_per_call))
            futures.append(executor.submit(ask_deepseek, combined, qa_per_call))
            futures.append(executor.submit(ask_gemini, combined, qa_per_call))
            futures.append(executor.submit(ask_huggingface, combined, qa_per_call))
            for future in as_completed(futures):
                res = future.result()
                if res: all_raws.append(res)
        entries = []
        for raw in all_raws: entries.extend(parse_qa_text(raw))
        print(f"📝 Total entries: {len(entries)}")
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
        else: print("⚠️ No entries this cycle.")
        elapsed = time.time() - start_cycle
        sleep_time = max(5, 15 - elapsed)
        print(f"⏳ Sleeping {sleep_time:.1f}s...")
        time.sleep(sleep_time)
    print("🏁 Non-stop run completed.")

if __name__ == "__main__":
    main()
