import requests
import json

def query_all_medical_apis():
    out = []
    # 6. NCBI PubMed
    try:
        ncbi = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                            params={"db": "pubmed", "term": "cancer immunotherapy", "retmax": 5})
        out.append("=== NCBI ===\n" + ncbi.text)
    except:
        pass
    # 7. OpenFDA
    try:
        fda = requests.get("https://api.fda.gov/drug/label.json",
                           params={"search": "cancer", "limit": 3}).json()
        out.append("=== OpenFDA ===\n" + str(fda))
    except:
        pass
    # 8. ClinicalTrials.gov
    try:
        ct = requests.get("https://clinicaltrials.gov/api/v2/studies",
                          params={"query.term": "cancer", "pageSize": 3}).json()
        out.append("=== ClinicalTrials ===\n" + str(ct))
    except:
        pass
    # 9. Open Targets
    try:
        ot_query = {
            "query": """
            query disease($efoId: String!) {
                disease(efoId: $efoId) {
                    id name
                    associatedTargets { rows { target { id approvedSymbol } score } }
                }
            }""",
            "variables": {"efoId": "EFO_0000319"}
        }
        ot = requests.post("https://api.platform.opentargets.org/api/v4/graphql", json=ot_query).json()
        out.append("=== Open Targets ===\n" + str(ot))
    except:
        pass
    # 10. Ensembl
    try:
        ens = requests.get("https://rest.ensembl.org/lookup/id/ENSG00000141510?expand=1",
                           headers={"Content-Type": "application/json"}).json()
        out.append("=== Ensembl ===\n" + str(ens))
    except:
        pass
    # 11. Reactome (Pathways)
    try:
        reactome = requests.get("https://reactome.org/ContentService/data/query/disease/cancer?species=Homo+sapiens&pageSize=5").json()
        out.append("=== Reactome ===\n" + json.dumps(reactome, indent=2)[:1500])
    except:
        pass
    # 12. KEGG (Pathways + Drugs)
    try:
        kegg = requests.get("http://rest.kegg.jp/list/pathway/hsa", timeout=10)
        if kegg.status_code == 200:
            out.append("=== KEGG ===\n" + kegg.text[:1000])
    except:
        pass
    # 13. UniProt (Proteins)
    try:
        uniprot = requests.get("https://rest.uniprot.org/uniprotkb/stream?format=json&query=%20(gene:BRCA2)%20AND%20(organism_id:9606)&size=3").json()
        out.append("=== UniProt ===\n" + str(uniprot))
    except:
        pass
    # 14. STRING (Protein Interactions)
    try:
        string_api = requests.get("https://string-db.org/api/json/network?identifiers=TP53&species=9606").json()
        out.append("=== STRING ===\n" + str(string_api))
    except:
        pass
    return "\n\n".join(out)
