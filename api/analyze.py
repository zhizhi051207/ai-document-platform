import base64
import contextlib
import io
import json
import math
import re
import unicodedata
import warnings
from collections import Counter
from http.server import BaseHTTPRequestHandler

# 全局警告抑制 - 防止pdfplumber的警告
warnings.filterwarnings("ignore", message="CropBox missing from /Page.*")
warnings.simplefilter("ignore", category=UserWarning)  # 忽略所有用户警告
warnings.simplefilter("ignore", category=DeprecationWarning)  # 忽略弃用警告

import pdfplumber
from docx import Document
from snownlp import SnowNLP
import openai
import os
import sys

# OpenRouter API 配置
# 请用户在Vercel中设置环境变量 OPENROUTER_API_KEY
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("请设置环境变量 OPENROUTER_API_KEY。请在Vercel控制台设置：Settings → Environment Variables")
openai.api_key = OPENROUTER_API_KEY
openai.base_url = "https://openrouter.ai/api/v1"

# AI模型配置
AI_MODEL = "deepseek/deepseek-chat"

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
        # 重定向stderr以捕获底层库的直接输出，并确保警告被抑制
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 确保在此上下文中的所有警告都被抑制
            try:
                stderr_capture = io.StringIO()
                with contextlib.redirect_stderr(stderr_capture):
                    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                        pages = []
                        for page in pdf.pages:
                            try:
                                text = page.extract_text() or ""
                                pages.append(text)
                            except Exception as page_exc:
                                # 单页提取失败不影响其他页
                                pages.append(f"[页面提取错误: {str(page_exc)}]")
                        return "\n".join(pages)
            except Exception as pdf_exc:
                # PDF解析失败时返回空字符串
                return f"[PDF解析失败: {str(pdf_exc)}]"
    if lower.endswith(".docx") or lower.endswith(".doc"):
        try:
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as doc_exc:
            return f"[DOCX解析失败: {str(doc_exc)}]"
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""


# AI增强函数
def ai_summarize(text: str, max_sentences: int = 3) -> str:
    """使用AI生成摘要，失败时返回空字符串触发回退"""
    try:
        if len(text) < 100:
            return ""
        response = openai.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业文档分析助手，请用中文生成简洁准确的摘要。"},
                {"role": "user", "content": f"请用{max_sentences}句话总结以下内容：\n\n{text[:3000]}"}
            ],
            max_tokens=500,
            temperature=0.3,
            timeout=10
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""


def ai_extract_keywords(text: str, top_k: int = 8) -> list:
    """使用AI提取关键词，失败时返回空列表触发回退"""
    try:
        if len(text) < 200:
            return []
        response = openai.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业文档分析助手，请提取文档的关键词，每行一个。"},
                {"role": "user", "content": f"从以下内容中提取{top_k}个最重要的关键词，每行一个：\n\n{text[:4000]}"}
            ],
            max_tokens=300,
            temperature=0.2,
            timeout=10
        )
        content = response.choices[0].message.content.strip()
        keywords = [line.strip() for line in content.split('\n') if line.strip()]
        keywords = keywords[:top_k]
        return [{"term": kw, "score": 1.0 - (i * 0.05)} for i, kw in enumerate(keywords)]
    except Exception:
        return []


def ai_extract_conclusions(text: str, sections: list) -> list:
    """使用AI提取结论性观点，失败时返回空列表触发回退"""
    try:
        if len(text) < 300:
            return []
        section_titles = [s.get("title", "") for s in sections[:5]]
        response = openai.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业文档分析助手，请提取文档的主要结论或核心观点。"},
                {"role": "user", "content": f"基于以下文档内容和章节标题，提取3-5个主要结论或核心观点，每行一个：\n\n文档片段：{text[:3500]}\n\n章节标题：{', '.join(section_titles)}"}
            ],
            max_tokens=400,
            temperature=0.3,
            timeout=10
        )
        content = response.choices[0].message.content.strip()
        conclusions = [line.strip() for line in content.split('\n') if line.strip()]
        return conclusions[:5]
    except Exception:
        return []


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


def detect_document_type(text: str, keywords) -> str:
    """检测文档类型"""
    text_lower = text.lower()
    if any(word in text_lower for word in ["论文", "研究", "学术", "期刊", "会议"]):
        return "学术论文"
    elif any(word in text_lower for word in ["报告", "分析", "市场", "行业"]):
        return "行业报告"
    elif any(word in text_lower for word in ["技术", "开发", "代码", "api", "接口"]):
        return "技术文档"
    elif any(word in text_lower for word in ["合同", "协议", "法律", "条款"]):
        return "法律文书"
    else:
        return "通用文档"


def generate_recommended_charts(doc_type: str) -> list[str]:
    """根据文档类型推荐图表"""
    chart_map = {
        "学术论文": ["keyword_barchart", "conclusion_graph", "section_structure", "reference_distribution", "timeline"],
        "行业报告": ["keyword_barchart", "category_pie", "timeline", "entity_relationship", "sentiment_analysis"],
        "技术文档": ["code_distribution", "api_statistics", "section_structure", "entity_relationship", "named_entity"],
        "法律文书": ["section_structure", "named_entity", "keyword_barchart", "timeline"],
        "通用文档": ["keyword_barchart", "wordcloud", "category_pie", "section_structure"]
    }
    return chart_map.get(doc_type, ["keyword_barchart", "wordcloud", "category_pie"])


def generate_enhanced_charts(keywords, text: str, doc_type: str):
    """生成增强图表数据"""
    # 词云数据：从关键词生成
    wordcloud = []
    for i, item in enumerate(keywords[:20]):  # 取前20个关键词
        wordcloud.append({
            "text": item["term"],
            "value": int(item["score"] * 1000) + 10  # 转换为整数权重
        })
    
    # 饼图数据：基于关键词分类或模拟分类
    pie_categories = []
    if doc_type == "学术论文":
        pie_categories = ["研究背景", "方法设计", "实验结果", "讨论分析", "结论总结"]
    elif doc_type == "行业报告":
        pie_categories = ["市场分析", "竞争格局", "趋势预测", "风险提示", "建议措施"]
    elif doc_type == "技术文档":
        pie_categories = ["架构设计", "接口说明", "代码示例", "部署指南", "故障排除"]
    else:
        pie_categories = ["概述", "主体内容", "案例分析", "数据展示", "总结"]
    
    pie = []
    base_value = 100 // len(pie_categories)
    for i, category in enumerate(pie_categories):
        pie.append({
            "name": category,
            "value": base_value + (i * 2)  # 简单分布
        })
    
    # 时间线数据：尝试从文本提取日期，否则模拟
    timeline = []
    # 尝试匹配日期模式
    date_patterns = [
        r"\d{4}年\d{1,2}月\d{1,2}日",
        r"\d{4}-\d{1,2}-\d{1,2}",
        r"\d{4}/\d{1,2}/\d{1,2}",
        r"\d{4}年\d{1,2}月",
        r"\d{4}年"
    ]
    
    found_dates = []
    for pattern in date_patterns:
        dates = re.findall(pattern, text)
        found_dates.extend(dates)
    
    if found_dates:
        # 使用实际找到的日期
        unique_dates = list(set(found_dates))[:5]  # 最多5个
        for i, date in enumerate(unique_dates):
            timeline.append({
                "date": date,
                "event": f"文档中提到的关键时间点",
                "importance": 3 + i  # 1-5的重要性
            })
    else:
        # 模拟时间线
        timeline = [
            {"date": "2024-01", "event": "项目启动阶段", "importance": 3},
            {"date": "2024-03", "event": "初步研究完成", "importance": 4},
            {"date": "2024-06", "event": "核心成果发布", "importance": 5},
            {"date": "2024-09", "event": "应用推广阶段", "importance": 3},
            {"date": "2024-12", "event": "总结与展望", "importance": 4}
        ]
    
    # 实体网络：从关键词关系生成简单网络
    entity_network = {
        "nodes": [],
        "links": []
    }
    
    # 节点：前10个关键词
    top_keywords = keywords[:10]
    for i, item in enumerate(top_keywords):
        entity_network["nodes"].append({
            "id": item["term"],
            "group": i % 3 + 1  # 简单的分组
        })
    
    # 链接：创建关键词之间的简单关系
    for i in range(len(top_keywords) - 1):
        entity_network["links"].append({
            "source": top_keywords[i]["term"],
            "target": top_keywords[i + 1]["term"],
            "value": 1 + (i % 3)  # 1-3的权重
        })
    
    return {
        "wordcloud": wordcloud,
        "pie": pie,
        "timeline": timeline,
        "entity_network": entity_network
    }


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
            # 尝试AI增强，失败则回退到传统算法
            ai_summary = ai_summarize(content, max_sentences=2)
            summary = normalize_text(ai_summary) if ai_summary else normalize_text(summarize(content, max_sentences=2))
            
            ai_keywords = ai_extract_keywords(content, top_k=8)
            keywords = ai_keywords if ai_keywords else extract_keywords(content, top_k=8)
            excerpt = content[:320] + ("..." if len(content) > 320 else "")
            sections.append({
                "title": section["title"],
                "excerpt": excerpt,
                "summary": summary,
                "keywords": keywords,
                "thinking": build_section_thinking(section["title"], summary, keywords),
            })
        # 文档级AI增强，失败则回退
        ai_keywords = ai_extract_keywords(text)
        keywords = ai_keywords if ai_keywords else extract_keywords(text)
        
        ai_summary = ai_summarize(text)
        summary = normalize_text(ai_summary) if ai_summary else normalize_text(summarize(text))
        
        ai_conclusions = ai_extract_conclusions(text, sections_raw)
        conclusions = [
            normalize_text(item)
            for item in (ai_conclusions if ai_conclusions else extract_conclusions(text, sections_raw))
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

    # 生成增强图表数据
    document_type = "通用文档"
    recommended_charts = []
    enhanced_charts_data = {
        "wordcloud": [],
        "pie": [],
        "timeline": [],
        "entity_network": {"nodes": [], "links": []}
    }
    
    if documents:
        # 使用第一个文档生成图表数据
        first_doc = documents[0]
        document_type = detect_document_type(first_doc["full_text"], first_doc["keywords"])
        recommended_charts = generate_recommended_charts(document_type)
        enhanced_charts_data = generate_enhanced_charts(
            first_doc["keywords"], 
            first_doc["full_text"], 
            document_type
        )

    return {
        "documents": cleaned_docs,
        "keyword_chart": keyword_chart,
        "conclusion_graph": conclusion_graph,
        "comparison": comparison,
        "conflicts": conflicts,
        "document_type": document_type,
        "recommended_charts": recommended_charts,
        "enhanced_charts": enhanced_charts_data,
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
