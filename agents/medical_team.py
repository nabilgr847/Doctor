import requests, json

def query_all_medical_apis():
    out = []
    try:
        ncbi = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                            params={"db": "pubmed", "term": "cancer immunotherapy", "retmax": 5}).text
        out.append("=== NCBI ===\n" + ncbi)
    except: pass
    try:
        fda = requests.get("https://api.fda.gov/drug/label.json", params={"search": "cancer", "limit": 3}).json()
        out.append("=== OpenFDA ===\n" + str(fda))
    except: pass
    try:
        ct = requests.get("https://clinicaltrials.gov/api/v2/studies", params={"query.term": "cancer", "pageSize": 3}).json()
        out.append("=== ClinicalTrials ===\n" + str(ct))
    except: pass
    try:
        ot_query = {
            "query": """query disease($efoId: String!) { disease(efoId: $efoId) { id name associatedTargets { rows { target { id approvedSymbol } score } } } }""",
            "variables": {"efoId": "EFO_0000319"}
        }
        ot = requests.post("https://api.platform.opentargets.org/api/v4/graphql", json=ot_query).json()
        out.append("=== Open Targets ===\n" + str(ot))
    except: pass
    try:
        ens = requests.get("https://rest.ensembl.org/lookup/id/ENSG00000141510?expand=1",
                           headers={"Content-Type": "application/json"}).json()
        out.append("=== Ensembl ===\n" + str(ens))
    except: pass
    # Reactome, KEGG, UNIPROT, STRING যদি রাখতে চাস, নিচের লাইনগুলো অ্যাক্টিভ কর
    # try: ...
    return "\n\n".join(out)
