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

# GROBID配置
try:
    from grobid_client import Client
    from grobid_client.models import ProcessForm
    from grobid_client.api.pdf import process_fulltext_document
    from grobid_client.types import File
    GROBID_AVAILABLE = True
except ImportError:
    GROBID_AVAILABLE = False
    print("警告: grobid_client未安装，将回退到pdfplumber")

# GROBID服务URL - 可以从环境变量配置
GROBID_SERVICE_URL = os.environ.get("GROBID_SERVICE_URL", "http://localhost:8070")
# 是否启用GROBID（默认启用，如果可用）
ENABLE_GROBID = os.environ.get("ENABLE_GROBID", "true").lower() == "true" and GROBID_AVAILABLE

# 导入GROBID解析器模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from grobid_parser import extract_with_grobid, grobid_to_text, parse_grobid_xml
    GROBID_PARSER_AVAILABLE = True
except ImportError:
    GROBID_PARSER_AVAILABLE = False
    print("警告: grobid_parser模块未找到，GROBID功能将受限")

# OpenRouter API 配置
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
openai.api_key = OPENROUTER_API_KEY
openai.base_url = "https://openrouter.ai/api/v1"

# AI模型配置
AI_MODEL = "anthropic/claude-sonnet-4.5"

def normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u3000", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    提取文档文本，优先使用GROBID（如果可用且启用）
    """
    lower = filename.lower()
    
    # 对于PDF文件，优先尝试GROBID
    if lower.endswith(".pdf") and ENABLE_GROBID and GROBID_PARSER_AVAILABLE:
        try:
            structured_data = extract_with_grobid(file_bytes, GROBID_SERVICE_URL)
            if structured_data:
                # 将结构化数据转换为纯文本格式
                return grobid_to_text(structured_data)
        except Exception as grobid_exc:
            print(f"GROBID提取失败，回退到pdfplumber: {grobid_exc}")
    
    # 原始提取逻辑（回退方案）
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


def extract_structured_data(file_bytes: bytes, filename: str) -> dict:
    """
    提取文档的结构化数据（使用GROBID）
    返回包含标题、作者、章节等结构化信息的字典
    """
    lower = filename.lower()
    
    if lower.endswith(".pdf") and ENABLE_GROBID and GROBID_PARSER_AVAILABLE:
        try:
            structured_data = extract_with_grobid(file_bytes, GROBID_SERVICE_URL)
            if structured_data:
                return structured_data
        except Exception as grobid_exc:
            print(f"GROBID结构化提取失败: {grobid_exc}")
    
    # 回退方案：使用普通文本提取
    text = extract_text(file_bytes, filename)
    return {
        "title": "",
        "authors": [],
        "abstract": "",
        "sections": [{"title": "全文", "content": text}],
        "references": [],
        "full_text": text,
        "structured_data": {}
    }


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


def process_document_with_ai(file_bytes: bytes, filename: str) -> dict:
    """
    使用GROBID和AI处理单个文档
    返回与原有文档结构兼容的分析结果
    """
    try:
        # 提取结构化数据
        structured_data = extract_structured_data(file_bytes, filename)
        
        # 提取纯文本（用于回退和兼容）
        raw_text = extract_text(file_bytes, filename) or ""
        text = normalize_text(raw_text)
        
        # AI全面分析
        ai_analysis = ai_analyze_document(structured_data, text)
        
        # 构建章节数据（使用结构化数据中的章节或回退到传统分割）
        sections_raw = []
        if structured_data.get("sections"):
            # 使用GROBID提取的章节
            for section in structured_data["sections"]:
                sections_raw.append({
                    "title": section.get("title", ""),
                    "content": section.get("content", "")
                })
        else:
            # 回退到传统章节分割
            sections_raw = split_sections(raw_text)
        
        sections = []
        for section in sections_raw:
            content = section["content"]
            # 使用AI分析章节内容
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
        
        # 使用AI分析的结果作为主要输出
        summary = ai_analysis.get("summary", "")
        keywords = ai_analysis.get("keywords", [])
        conclusions = ai_analysis.get("conclusions", [])
        document_type = ai_analysis.get("document_type", "通用文档")
        
        # 如果AI分析结果为空，使用原有AI函数
        if not summary:
            summary = ai_summarize(text)
        if not keywords:
            keywords = ai_extract_keywords(text)
        if not conclusions:
            conclusions = ai_extract_conclusions(text, sections_raw)
        
        # 转换关键词格式（如果需要）
        keyword_list = []
        if keywords and isinstance(keywords, list):
            for i, kw in enumerate(keywords[:10]):
                if isinstance(kw, dict) and "term" in kw:
                    keyword_list.append(kw)
                else:
                    keyword_list.append({
                        "term": str(kw),
                        "score": 1.0 - (i * 0.1)
                    })
        
        return {
            "name": filename,
            "word_count": len(text),
            "summary": summary,
            "conclusions": conclusions,
            "keywords": keyword_list,
            "sections": sections,
            "full_text": text,
            "structured_info": ai_analysis.get("structured_info", {}),
            "document_type": document_type,
            "ai_enhanced": True
        }
        
    except Exception as e:
        print(f"AI处理文档失败，回退到传统处理: {e}")
        # 回退到传统处理流程
        raw_text = extract_text(file_bytes, filename) or ""
        text = normalize_text(raw_text)
        sections_raw = split_sections(raw_text)
        
        sections = []
        for section in sections_raw:
            content = section["content"]
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
        
        return {
            "name": filename,
            "word_count": len(text),
            "summary": summary,
            "conclusions": conclusions,
            "keywords": keywords,
            "sections": sections,
            "full_text": text,
            "structured_info": {},
            "document_type": detect_document_type(text, keywords),
            "ai_enhanced": False
        }


def build_response(payload):
    documents = []
    
    # 处理每个文档
    for file in payload.get("files", []):
        try:
            file_bytes = base64.b64decode(file["base64"])
            # 使用AI增强处理文档
            doc_result = process_document_with_ai(file_bytes, file["name"])
            documents.append(doc_result)
        except Exception as e:
            print(f"处理文档 {file.get('name', 'unknown')} 失败: {e}")
            # 添加一个空文档占位符，防止完全失败
            documents.append({
                "name": file.get("name", "error"),
                "word_count": 0,
                "summary": "文档处理失败",
                "conclusions": [],
                "keywords": [],
                "sections": [],
                "full_text": "",
                "structured_info": {},
                "document_type": "未知",
                "ai_enhanced": False
            })
    
    # 关键词图表（使用第一个文档）
    keyword_chart = []
    if documents and documents[0].get("keywords"):
        for item in documents[0]["keywords"]:
            if isinstance(item, dict) and "term" in item:
                keyword_chart.append({"name": item["term"], "value": round(item.get("score", 1) * 100, 2)})
            elif isinstance(item, str):
                keyword_chart.append({"name": item, "value": 100})
    
    # 结论图
    conclusion_graph = build_conclusion_graph(
        documents[0].get("keywords", []), 
        documents[0].get("full_text", "")
    ) if documents else {
        "nodes": [],
        "edges": [],
    }
    
    # 文档比较
    comparison = compare_documents(documents)
    
    # 冲突检测
    conflicts = detect_conflicts(documents)
    
    # 清理文档数据（移除full_text字段）
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
        doc_type = first_doc.get("document_type", "通用文档")
        if doc_type and doc_type != "通用文档":
            document_type = doc_type
        else:
            # 回退到自动检测
            document_type = detect_document_type(first_doc.get("full_text", ""), first_doc.get("keywords", []))
        
        recommended_charts = generate_recommended_charts(document_type)
        
        # 生成增强图表
        keywords_for_charts = first_doc.get("keywords", [])
        # 转换关键词格式
        keyword_terms = []
        for kw in keywords_for_charts:
            if isinstance(kw, dict) and "term" in kw:
                keyword_terms.append(kw["term"])
            elif isinstance(kw, str):
                keyword_terms.append(kw)
        
        enhanced_charts_data = generate_enhanced_charts(
            keywords_for_charts, 
            first_doc.get("full_text", ""), 
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
        "ai_enhanced": any(doc.get("ai_enhanced", False) for doc in documents)
    }


def ai_analyze_document(structured_data: dict, text: str) -> dict:
    """
    使用AI对文档进行全面分析，结合GROBID结构化数据
    
    Args:
        structured_data: GROBID提取的结构化数据
        text: 文档纯文本
    
    Returns:
        包含所有AI生成分析结果的字典
    """
    try:
        # 准备AI提示词
        title = structured_data.get("title", "")
        authors = structured_data.get("authors", [])
        abstract = structured_data.get("abstract", "")
        sections = structured_data.get("sections", [])
        references = structured_data.get("references", [])
        
        # 构建结构化信息描述
        structured_info = ""
        if title:
            structured_info += f"标题: {title}\n"
        if authors:
            structured_info += f"作者: {', '.join(authors)}\n"
        if abstract:
            structured_info += f"摘要: {abstract}\n"
        if sections:
            structured_info += f"文档包含 {len(sections)} 个主要章节。\n"
        
        # 构建章节内容（用于AI分析）
        section_texts = []
        for i, section in enumerate(sections[:10]):  # 限制前10个章节
            section_texts.append(f"章节 {i+1}: {section.get('title', '')}\n{section.get('content', '')[:500]}")
        
        # 完整的AI分析提示词
        system_prompt = """你是一个专业文档分析助手。请基于提供的文档内容，生成全面、准确的分析结果。
请用中文回答，保持专业性和客观性。"""

        user_prompt = f"""请分析以下文档：

{structured_info}

文档全文（摘要）：
{text[:5000]}

章节概览：
{"\n".join(section_texts)}

请提供以下分析结果（请直接给出结果，不要添加额外解释）：

1. 文档摘要（3-5句话）：
2. 关键词（8-10个，按重要性排序）：
3. 主要章节概要（列出主要章节及其核心内容）：
4. 核心结论/观点（5-7个）：
5. 文档类型（如学术论文、技术报告、商业文档等）：
6. 分析难度（简单/中等/复杂）：
7. 建议的可视化图表类型（如词云、时间线、关系图等）：
"""
        
        # 调用AI
        response = openai.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1500,
            temperature=0.3,
            timeout=15
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # 解析AI响应（简单解析逻辑）
        # 这里可以改进为更智能的解析，暂时先返回原始响应
        # 同时尝试提取各个部分
        
        # 尝试提取摘要（第一项）
        summary = ""
        if "1. 文档摘要" in ai_response:
            parts = ai_response.split("1. 文档摘要")
            if len(parts) > 1:
                summary_part = parts[1].split("2. 关键词")[0] if "2. 关键词" in parts[1] else parts[1]
                summary = summary_part.strip().strip("：").strip()
        
        # 如果无法解析，使用AI生成的摘要函数作为回退
        if not summary:
            summary = ai_summarize(text)
        
        # 提取关键词
        keywords = ai_extract_keywords(text)
        
        # 提取结论
        conclusions = ai_extract_conclusions(text, sections)
        
        # 检测文档类型（需要关键词参数）
        document_type = detect_document_type(text, keywords)
        
        # 生成图表建议
        recommended_charts = generate_recommended_charts(document_type)
        
        # 生成增强图表数据
        enhanced_charts_data = generate_enhanced_charts(
            keywords, 
            text, 
            document_type
        )
        
        return {
            "summary": summary,
            "keywords": keywords,
            "conclusions": conclusions,
            "document_type": document_type,
            "recommended_charts": recommended_charts,
            "enhanced_charts": enhanced_charts_data,
            "ai_full_response": ai_response,  # 保留完整AI响应用于调试
            "structured_info": {
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "section_count": len(sections),
                "reference_count": len(references)
            }
        }
        
    except Exception as e:
        print(f"AI全面分析失败: {e}")
        # 回退到原有的AI函数
        return {
            "summary": ai_summarize(text),
            "keywords": ai_extract_keywords(text),
            "conclusions": ai_extract_conclusions(text, []),
            "document_type": detect_document_type(text, []),
            "recommended_charts": [],
            "enhanced_charts": {},
            "ai_full_response": "",
            "structured_info": {}
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
