import base64
import io
import json
import math
import re
import unicodedata
from collections import Counter
from http.server import BaseHTTPRequestHandler

import pdfplumber
from docx import Document
from snownlp import SnowNLP


def normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u3000", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


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


def tokenize(text: str):
    tokens = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]+", text.lower())
    cleaned = [token for token in tokens if len(token) > 1]
    return cleaned


def compute_tfidf(doc_tokens):
    doc_count = len(doc_tokens)
    df = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))

    tfidf_docs = []
    for tokens in doc_tokens:
        tf = Counter(tokens)
        tfidf = {}
        for term, freq in tf.items():
            idf = math.log((doc_count + 1) / (df[term] + 1)) + 1
            tfidf[term] = freq * idf
        tfidf_docs.append(tfidf)
    return tfidf_docs


def summarize(text: str, max_sentences: int = 3):
    sentences = re.split(r"[。！？.!?]\s*", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return "暂无摘要"
    tokenized = [tokenize(sentence) for sentence in sentences]
    word_freq = Counter([token for sent in tokenized for token in sent])
    scores = []
    for sentence, tokens in zip(sentences, tokenized):
        if not tokens:
            continue
        score = sum(word_freq[token] for token in tokens) / len(tokens)
        scores.append((sentence, score))
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    selected = [sent for sent, _ in ranked[:max_sentences]]
    return "。".join(selected) + "。"


def split_sections(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections = []
    current_title = "概览"
    current_content = []
    heading_pattern = re.compile(r"^(\d+(?:\.\d+)*\s+.+|第.+章|摘要|结论|Conclusion|Abstract)$", re.I)

    for line in lines:
        if heading_pattern.match(line):
            if current_content:
                joined = normalize_text(" ".join(current_content))
                sections.append({
                    "title": normalize_text(current_title) or current_title,
                    "content": joined[:180] + ("..." if len(joined) > 180 else ""),
                })
            current_title = line
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        joined = normalize_text(" ".join(current_content))
        sections.append({
            "title": normalize_text(current_title) or current_title,
            "content": joined[:180] + ("..." if len(joined) > 180 else ""),
        })

    return sections


def extract_keywords(text: str, top_k: int = 8):
    tokens = tokenize(text)
    if not tokens:
        return []
    tf = Counter(tokens)
    total = sum(tf.values())
    ranked = sorted(tf.items(), key=lambda x: x[1], reverse=True)
    keywords = []
    for term, freq in ranked[:top_k]:
        keywords.append({"term": term, "score": round(freq / total, 4)})
    return keywords


def extract_conclusions(text: str, sections):
    for section in sections:
        if "结论" in section["title"] or "conclusion" in section["title"].lower():
            return [s for s in re.split(r"[。！？.!?]\s*", section["content"]) if s][:3]
    sentences = [s.strip() for s in re.split(r"[。！？.!?]\s*", text) if len(s.strip()) > 10]
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


def cosine_similarity(vec_a, vec_b):
    intersection = set(vec_a.keys()) | set(vec_b.keys())
    numerator = sum(vec_a.get(term, 0) * vec_b.get(term, 0) for term in intersection)
    denom_a = math.sqrt(sum(value * value for value in vec_a.values()))
    denom_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return numerator / (denom_a * denom_b)


def compare_documents(documents):
    texts = [doc["full_text"] for doc in documents]
    if not texts:
        return {"similarity": [], "overlap_keywords": [], "unique_keywords": {}}

    tokenized_docs = [tokenize(text) for text in texts]
    tfidf_docs = compute_tfidf(tokenized_docs)

    sim_matrix = []
    for vec_a in tfidf_docs:
        row = [round(cosine_similarity(vec_a, vec_b), 4) for vec_b in tfidf_docs]
        sim_matrix.append(row)

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


def build_response(payload):
    documents = []
    for file in payload.get("files", []):
        file_bytes = base64.b64decode(file["base64"])
        raw_text = extract_text(file_bytes, file["name"]) or ""
        text = normalize_text(raw_text)
        sections = split_sections(text)
        keywords = extract_keywords(text)
        summary = normalize_text(summarize(text))
        conclusions = [
            normalize_text(item)
            for item in extract_conclusions(text, sections)
            if normalize_text(item)
        ]
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
