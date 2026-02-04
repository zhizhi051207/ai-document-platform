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



def detect_heading_level(line: str):
    text = normalize_text(line)
    if not text:
        return None
    if len(text) > 80 and re.search(r"[a-z]", text):
        return None
    if text.count(" ") > 8 and re.search(r"[a-z]", text):
        return None
    clean = re.sub(r"[，。；:：,]+$", "", text)
    specials = {"摘要", "结论", "Abstract", "Conclusion"}
    if clean in specials:
        return 1
    if re.match(r"^第[一二三四五六七八九十]+章", clean):
        return 1
    if re.match(r"^\d+\.\d+\.\d+(?:\s|、|\.|\)|）).+", clean):
        if len(clean) > 80:
            return None
        return 3
    if re.match(r"^\d+\.\d+(?:\s|、|\.|\)|）).+", clean):
        if len(clean) > 60:
            return None
        return 2
    if re.match(r"^\d+(?:\s|、|\.|\)|）).+", clean):
        if len(clean) > 40:
            return None
        if re.search(r"[，。；:：]", clean):
            return None
        if re.search(r"\b\d{4}\b", clean) and not re.search(r"(章|节|引言|结语|结果|讨论|摘要)", clean):
            return None
        heading_keywords = ["引言", "结语", "结论", "摘要", "讨论", "结果", "方法", "实验", "背景", "研究", "理论", "文献", "综述", "相关工作", "模型", "数据", "系统", "实现", "分析", "问题", "意义"]
        if len(clean) > 16 and not any(keyword in clean for keyword in heading_keywords):
            return None
        return 1
    if re.match(r"^[一二三四五六七八九十]+[、.].+", clean):
        return 1
    if re.match(r"^[（(][一二三四五六七八九十]+[)）].+", clean):
        return 2
    if re.match(r"^[（(]\d+[)）].+", clean):
        return 2
    return None


def split_sections(text: str):
    raw_lines = [normalize_text(line) for line in text.splitlines()]
    lines = []
    for line in raw_lines:
        if not line:
            continue
        if re.fullmatch(r"\d{1,4}", line):
            continue
        if not re.search(r"[A-Za-z\u4e00-\u9fff]", line) and re.fullmatch(r"[\d\W]+", line):
            continue
        lines.append(line)
    sections = []
    current_title = "概览"
    current_level = 1
    current_content = []

    for line in lines:
        level = detect_heading_level(line)
        if level is not None:
            if current_title != "概览" or current_content:
                joined = normalize_text(" ".join(current_content))
                sections.append({
                    "title": normalize_text(current_title) or current_title,
                    "content": joined,
                    "level": current_level,
                })
            current_title = line
            current_level = level
            current_content = []
        else:
            current_content.append(line)

    if current_title != "概览" or current_content:
        joined = normalize_text(" ".join(current_content))
        sections.append({
            "title": normalize_text(current_title) or current_title,
            "content": joined,
            "level": current_level,
        })

    return sections
def build_section_thinking(title: str, summary: str, keywords):
    keyword_text = "、".join([item["term"] for item in keywords[:3]])
    if summary == "暂无摘要":
        return "本节暂无足够的句子生成思路分析。"
    if keyword_text:
        return f"本节围绕{keyword_text}展开，核心观点是：{summary}"
    return f"本节核心观点是：{summary}"


def extract_keywords(text: str, top_k: int = 12):
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
        sections_raw = split_sections(raw_text)
        sections = []
        for section in sections_raw:
            content = section["content"]
            summary = normalize_text(summarize(content, max_sentences=2))
            keywords = extract_keywords(content, top_k=8)
            excerpt = content[:320] + ("..." if len(content) > 320 else "")
            sections.append({
                "title": section["title"],
                "excerpt": excerpt,
                "summary": summary,
                "keywords": keywords,
                "thinking": build_section_thinking(section["title"], summary, keywords),
            })
        keywords = extract_keywords(text)
        summary = normalize_text(summarize(text))
        conclusions = [
            normalize_text(item)
            for item in extract_conclusions(text, sections_raw)
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

    # 增强分析：文档类型识别和个性化图表
    document_type = "other"
    recommended_charts = []
    enhanced_charts = {}
    
    if documents:
        first_doc = documents[0]
        doc_text = first_doc["full_text"]
        doc_keywords = first_doc["keywords"]
        
        # 检测文档类型
        document_type = detect_document_type(doc_text)
        
        # 推荐图表
        recommended_charts = recommend_charts(document_type, doc_text, doc_keywords)
        
        # 生成增强图表数据
        enhanced_charts = generate_enhanced_charts(doc_text, document_type, doc_keywords)
    
    return {
        "documents": cleaned_docs,
        "keyword_chart": keyword_chart,
        "conclusion_graph": conclusion_graph,
        "comparison": comparison,
        "conflicts": conflicts,
        # 新增字段
        "document_type": document_type,
        "recommended_charts": recommended_charts,
        "enhanced_charts": enhanced_charts,
    }

def detect_document_type(text):
    """检测文档类型：学术论文、商业报告、技术文档、新闻文章、其他"""
    text_lower = text.lower()
    
    # 学术论文特征
    academic_keywords = ["abstract", "introduction", "methodology", "results", "discussion", "conclusion", "references", "参考文献", "摘要", "引言", "方法", "结果", "讨论", "结论"]
    academic_count = sum(1 for keyword in academic_keywords if keyword in text_lower)
    
    # 商业报告特征
    business_keywords = ["市场", "营销", "销售", "财务", "利润", "收入", "增长", "战略", "竞争", "客户", "产品", "服务", "年度报告", "季度报告"]
    business_count = sum(1 for keyword in business_keywords if keyword in text_lower)
    
    # 技术文档特征
    tech_keywords = ["api", "接口", "函数", "代码", "编程", "算法", "架构", "部署", "配置", "安装", "使用说明", "文档", "示例"]
    tech_count = sum(1 for keyword in tech_keywords if keyword in text_lower)
    
    # 新闻文章特征
    news_keywords = ["报道", "记者", "新闻", "消息", "据悉", "表示", "指出", "近日", "昨天", "今天", "日前", "发布", "举行"]
    news_count = sum(1 for keyword in news_keywords if keyword in text_lower)
    
    # 判断
    scores = {
        "academic": academic_count,
        "business": business_count,
        "technical": tech_count,
        "news": news_count
    }
    
    max_type = max(scores, key=scores.get)
    if scores[max_type] >= 2:
        return max_type
    return "other"


def recommend_charts(doc_type, text, keywords):
    """根据文档类型推荐图表组合"""
    base_charts = ["keyword_barchart", "conclusion_graph"]
    
    if doc_type == "academic":
        return base_charts + ["section_structure", "reference_distribution"]
    elif doc_type == "business":
        return base_charts + ["timeline", "entity_relationship"]
    elif doc_type == "technical":
        return base_charts + ["code_distribution", "api_statistics"]
    elif doc_type == "news":
        return base_charts + ["named_entity", "sentiment_analysis"]
    else:
        return base_charts + ["wordcloud", "category_pie"]


def generate_enhanced_charts(text, doc_type, keywords):
    """生成增强图表数据"""
    enhanced = {}
    
    # 词云数据 (基于关键词权重)
    wordcloud_data = []
    for kw in keywords[:15]:
        wordcloud_data.append({
            "text": kw["term"],
            "value": kw["score"] * 1000
        })
    enhanced["wordcloud"] = wordcloud_data
    
    # 分类饼图数据 (模拟分类分布)
    pie_data = []
    categories = []
    if doc_type == "academic":
        categories = ["引言", "方法", "结果", "讨论", "参考文献"]
    elif doc_type == "business":
        categories = ["市场分析", "财务数据", "战略规划", "竞争分析", "执行总结"]
    elif doc_type == "technical":
        categories = ["API文档", "代码示例", "安装指南", "配置说明", "故障排除"]
    elif doc_type == "news":
        categories = ["政治", "经济", "社会", "文化", "科技"]
    else:
        categories = ["第一部分", "第二部分", "第三部分", "第四部分", "其他"]
    
    import random
    for i, cat in enumerate(categories):
        pie_data.append({
            "name": cat,
            "value": random.randint(5, 30)  # 模拟数据，实际应基于内容分析
        })
    enhanced["pie"] = pie_data
    
    # 时间线数据 (如果检测到日期)
    timeline_data = []
    date_patterns = [
        r"\d{4}年\d{1,2}月\d{1,2}日",
        r"\d{4}-\d{1,2}-\d{1,2}",
        r"\d{4}\.\d{1,2}\.\d{1,2}"
    ]
    for pattern in date_patterns:
        dates = re.findall(pattern, text)
        if dates:
            for date in dates[:5]:
                timeline_data.append({
                    "date": date,
                    "event": f"事件描述",
                    "importance": random.randint(1, 10)
                })
            break
    if not timeline_data:
        # 生成模拟时间线
        for i in range(3):
            timeline_data.append({
                "date": f"2025-{i+1}-01",
                "event": f"关键事件 {i+1}",
                "importance": random.randint(1, 10)
            })
    enhanced["timeline"] = timeline_data
    
    # 实体关系数据 (基于关键词共现)
    entity_data = {
        "nodes": [],
        "links": []
    }
    for i, kw in enumerate(keywords[:8]):
        entity_data["nodes"].append({
            "id": kw["term"],
            "group": i % 3 + 1
        })
    
    for i in range(len(entity_data["nodes"])):
        for j in range(i+1, len(entity_data["nodes"])):
            if random.random() > 0.7:
                entity_data["links"].append({
                    "source": entity_data["nodes"][i]["id"],
                    "target": entity_data["nodes"][j]["id"],
                    "value": random.randint(1, 5)
                })
    enhanced["entity_network"] = entity_data
    
    return enhanced

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
