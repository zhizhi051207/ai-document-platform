import base64
import io
import json
import re
from http.server import BaseHTTPRequestHandler

import pdfplumber
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from snownlp import SnowNLP


def extract_text(file_bytes: bytes, filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)
    if lower.endswith(".docx") or lower.endswith(".doc"):
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs])
    return file_bytes.decode("utf-8", errors="ignore")


def split_sections(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections = []
    current_title = "概览"
    current_content = []
    heading_pattern = re.compile(r"^(\d+(?:\.\d+)*\s+.+|第.+章|摘要|结论|Conclusion|Abstract)$", re.I)

    for line in lines:
        if heading_pattern.match(line):
            if current_content:
                sections.append({
                    "title": current_title,
                    "content": " ".join(current_content)[:180] + ("..." if len(" ".join(current_content)) > 180 else ""),
                })
            current_title = line
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections.append({
            "title": current_title,
            "content": " ".join(current_content)[:180] + ("..." if len(" ".join(current_content)) > 180 else ""),
        })

    return sections


def extract_keywords(text: str, top_k: int = 8):
    vectorizer = TfidfVectorizer(max_features=2000, stop_words=None)
    tfidf = vectorizer.fit_transform([text])
    scores = tfidf.toarray()[0]
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return [{"term": term, "score": round(score, 4)} for term, score in ranked[:top_k]]


def summarize(text: str, max_sentences: int = 3):
    sentences = re.split(r"[。！？.!?]\s*", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return "暂无摘要"
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)
    scores = tfidf.sum(axis=1).A1
    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    selected = [sent for sent, _ in ranked[:max_sentences]]
    return "。".join(selected) + "。"


def extract_conclusions(text: str, sections):
    for section in sections:
        if "结论" in section["title"] or "conclusion" in section["title"].lower():
            return re.split(r"[。！？.!?]\s*", section["content"])[:3]
    sentences = re.split(r"[。！？.!?]\s*", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences[-3:] if len(sentences) >= 3 else sentences


def build_conclusion_graph(keywords, text):
    nodes = [item["term"] for item in keywords]
    edges = []
    sentences = re.split(r"[。！？.!?]\s*", text)
    for i, source in enumerate(nodes):
        for target in nodes[i + 1 :]:
            weight = 0
            for sentence in sentences:
                if source in sentence and target in sentence:
                    weight += 1
            if weight > 0:
                edges.append({"source": source, "target": target, "weight": weight})
    edges = sorted(edges, key=lambda x: x["weight"], reverse=True)[:12]
    return {"nodes": nodes, "edges": edges}


def detect_conflicts(documents):
    if len(documents) < 2:
        return []
    conflicts = []
    doc_a = documents[0]
    doc_b = documents[1]
    text_a = doc_a["full_text"]
    text_b = doc_b["full_text"]
    keywords = set([k["term"] for k in doc_a["keywords"]]) & set([k["term"] for k in doc_b["keywords"]])

    for keyword in keywords:
        sentences_a = [s for s in re.split(r"[。！？.!?]\s*", text_a) if keyword in s]
        sentences_b = [s for s in re.split(r"[。！？.!?]\s*", text_b) if keyword in s]
        if not sentences_a or not sentences_b:
            continue
        sentence_a = sentences_a[0]
        sentence_b = sentences_b[0]
        sentiment_a = SnowNLP(sentence_a).sentiments
        sentiment_b = SnowNLP(sentence_b).sentiments
        if abs(sentiment_a - sentiment_b) >= 0.4:
            conflicts.append({
                "topic": keyword,
                "doc_a": sentence_a,
                "doc_b": sentence_b,
                "sentiment_a": sentiment_a,
                "sentiment_b": sentiment_b,
            })
    return conflicts


def compare_documents(documents):
    texts = [doc["full_text"] for doc in documents]
    if not texts:
        return {"similarity": [], "overlap_keywords": [], "unique_keywords": {}}
    vectorizer = TfidfVectorizer(max_features=3000)
    tfidf = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf).tolist()

    keyword_sets = [set([k["term"] for k in doc["keywords"]]) for doc in documents]
    overlap = set.intersection(*keyword_sets) if len(keyword_sets) > 1 else keyword_sets[0]
    unique = {}
    for idx, doc in enumerate(documents):
        others = set().union(*[s for i, s in enumerate(keyword_sets) if i != idx])
        unique[doc["name"]] = list(keyword_sets[idx] - others)

    return {
        "similarity": sim_matrix,
        "overlap_keywords": list(overlap) if overlap else [],
        "unique_keywords": unique,
    }


def build_response(payload):
    documents = []
    for file in payload.get("files", []):
        file_bytes = base64.b64decode(file["base64"])
        text = extract_text(file_bytes, file["name"]) or ""
        text = re.sub(r"\s+", " ", text)
        sections = split_sections(text)
        keywords = extract_keywords(text)
        summary = summarize(text)
        conclusions = extract_conclusions(text, sections)
        documents.append({
            "name": file["name"],
            "word_count": len(text),
            "summary": summary,
            "conclusions": conclusions,
            "keywords": keywords,
            "sections": sections,
            "full_text": text,
        })

    keyword_chart = []
    if documents:
        for item in documents[0]["keywords"]:
            keyword_chart.append({"name": item["term"], "value": round(item["score"] * 100, 2)})

    conclusion_graph = build_conclusion_graph(documents[0]["keywords"], documents[0]["full_text"]) if documents else {
        "nodes": [],
        "edges": [],
    }

    comparison = compare_documents(documents)
    conflicts = detect_conflicts(documents)

    cleaned_docs = []
    for doc in documents:
        doc_copy = {k: v for k, v in doc.items() if k != "full_text"}
        cleaned_docs.append(doc_copy)

    return {
        "documents": cleaned_docs,
        "keyword_chart": keyword_chart,
        "conclusion_graph": conclusion_graph,
        "comparison": comparison,
        "conflicts": conflicts,
    }


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
            response = build_response(payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            )

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
