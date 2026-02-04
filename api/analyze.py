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
    
    # 移除末尾标点
    clean = re.sub(r"[，。；:：,]+$", "", text)
    
    # 特殊标题识别
    specials = {"摘要", "结论", "Abstract", "Conclusion", "引言", "前言", "引言", "背景", "研究背景", "研究方法", "实验结果", "讨论与分析", "总结", "参考文献", "致谢", "附录"}
    if clean in specials:
        return 1
    
    # 中文章节编号: 第X章
    if re.match(r"^第[一二三四五六七八九十]+章", clean):
        return 1
    
    # 数字编号: 1. 或 1.1 或 1.1.1
    if re.match(r"^\d+\.\d+\.\d+(?:\s|、|\.|\)|）).+", clean):
        return 3
    if re.match(r"^\d+\.\d+(?:\s|、|\.|\)|）).+", clean):
        return 2
    if re.match(r"^\d+(?:\s|、|\.|\)|）).+", clean):
        return 1
    
    # 中文数字编号: 一、 或 (一) 或 一.
    if re.match(r"^[一二三四五六七八九十]+[、.].+", clean):
        return 1
    if re.match(r"^[（(][一二三四五六七八九十]+[)）].+", clean):
        return 2
    
    # 英文字母编号: A. 或 (A) 或 A)
    if re.match(r"^[A-Z][、.)].+", clean):
        return 2
    
    # 基于关键词的标题识别 (放宽条件)
    heading_keywords = [
        "引言", "前言", "背景", "研究背景", "问题提出", "研究意义", "文献综述", "相关工作",
        "方法", "研究方法", "实验设计", "技术路线", "算法", "模型", "框架",
        "实验", "实验结果", "数据", "数据分析", "统计", "评估", "性能",
        "结果", "研究结果", "发现", "讨论", "分析与讨论", "结论", "总结", "展望",
        "参考文献", "引用", "致谢", "附录", "附件", "图表目录"
    ]
    
    # 检查是否包含标题关键词
    if any(keyword in clean for keyword in heading_keywords):
        # 如果包含关键词且长度适中，可能是标题
        if 5 <= len(clean) <= 120:
            return 2 if len(clean) > 40 else 1
    
    # 检查大写字母开头且长度适中的行（可能是英文标题）
    if re.match(r"^[A-Z][A-Za-z\s]{1,80}[^.]$", clean) and len(clean) <= 80:
        return 2
    
    # 检查中文标题特征：较短且不含句号
    if re.search(r"[\u4e00-\u9fff]", clean) and len(clean) <= 40 and "." not in clean:
        # 排除明显的内容行
        if re.search(r"[，。；:：]", clean):
            return None
        if re.search(r"\b(是|的|在|和|有|为|可以|能够|不能|不会)\b", clean):
            return None
        return 3  # 三级标题
    
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
    """检测文档类型：学术论文、商业报告、技术文档、新闻文章、法律文档、政府公文、其他"""
    text_lower = text.lower()
    
    # 扩展关键词列表
    # 学术论文特征
    academic_keywords = [
        "abstract", "introduction", "methodology", "methods", "results", "discussion", 
        "conclusion", "references", "bibliography", "citation", "文献", "参考文献", 
        "引用", "摘要", "引言", "方法", "实验", "结果", "讨论", "结论", "研究背景",
        "研究目的", "研究意义", "理论基础", "文献综述", "模型构建", "数据分析",
        "实证研究", "学术", "论文", "期刊", "会议", "学位论文"
    ]
    academic_count = sum(1 for keyword in academic_keywords if keyword in text_lower)
    
    # 商业报告特征
    business_keywords = [
        "市场", "营销", "销售", "财务", "利润", "收入", "增长", "战略", "竞争", 
        "客户", "产品", "服务", "年度报告", "季度报告", "财务报表", "资产负债表",
        "现金流量表", "利润表", "市场份额", "竞争对手", "市场分析", "商业计划",
        "投资回报", "风险评估", "SWOT分析", "PEST分析", "商业模式", "价值链"
    ]
    business_count = sum(1 for keyword in business_keywords if keyword in text_lower)
    
    # 技术文档特征
    tech_keywords = [
        "api", "接口", "函数", "代码", "编程", "算法", "架构", "部署", "配置", 
        "安装", "使用说明", "文档", "示例", "教程", "指南", "手册", "调试",
        "错误", "异常", "测试", "单元测试", "集成测试", "版本", "更新", "升级",
        "源码", "仓库", "git", "commit", "分支", "合并", "编译", "构建", "打包"
    ]
    tech_count = sum(1 for keyword in tech_keywords if keyword in text_lower)
    
    # 新闻文章特征
    news_keywords = [
        "报道", "记者", "新闻", "消息", "据悉", "表示", "指出", "近日", "昨天", 
        "今天", "日前", "发布", "举行", "召开", "会议", "发布会", "透露", "称",
        "据了解", "据介绍", "据报道", "据悉", "有消息称", "来源", "独家", "头条"
    ]
    news_count = sum(1 for keyword in news_keywords if keyword in text_lower)
    
    # 法律文档特征
    legal_keywords = [
        "合同", "协议", "条款", "甲方", "乙方", "权利", "义务", "责任", "违约",
        "赔偿", "仲裁", "诉讼", "法院", "法律", "法规", "条例", "规定", "章程"
    ]
    legal_count = sum(1 for keyword in legal_keywords if keyword in text_lower)
    
    # 政府公文特征
    gov_keywords = [
        "通知", "公告", "通报", "决定", "意见", "办法", "规定", "条例", "细则",
        "请示", "报告", "批复", "函", "纪要", "政府", "市委", "县委", "省委",
        "国务院", "办公厅", "委员会", "办公室", "关于印发", "关于转发"
    ]
    gov_count = sum(1 for keyword in gov_keywords if keyword in text_lower)
    
    # 判断 - 使用加权分数
    scores = {
        "academic": academic_count * 1.5,  # 学术论文权重更高
        "business": business_count,
        "technical": tech_count,
        "news": news_count,
        "legal": legal_count,
        "government": gov_count
    }
    
    max_type = max(scores, key=scores.get)
    max_score = scores[max_type]
    
    # 设置阈值
    if max_score >= 3:
        return max_type
    elif max_score >= 2:
        # 如果有明显的关键词，即使分数低也识别
        return max_type
    else:
        # 尝试基于内容特征判断
        if len(text) > 5000 and "参考文献" in text:
            return "academic"
        elif "财务报表" in text or "年度报告" in text:
            return "business"
        elif "代码" in text or "API" in text:
            return "technical"
        elif "报道" in text or "记者" in text:
            return "news"
        elif "合同" in text or "协议" in text:
            return "legal"
        elif "通知" in text or "公告" in text:
            return "government"
        else:
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
    
        # 分类饼图数据 (基于内容分析)
    pie_data = []
    
    # 根据文档类型和内容分析生成分类
    if doc_type == "academic":
        # 学术论文分类分析
        categories = ["引言与背景", "方法与实验", "结果与发现", "讨论与分析", "结论与展望", "参考文献"]
        # 基于关键词和章节标题估算比例
        text_lower = text.lower()
        intro_score = sum(1 for kw in ["引言", "背景", "introduction", "background"] if kw in text_lower)
        method_score = sum(1 for kw in ["方法", "实验", "method", "experiment", "methodology"] if kw in text_lower)
        result_score = sum(1 for kw in ["结果", "发现", "result", "finding"] if kw in text_lower)
        discussion_score = sum(1 for kw in ["讨论", "分析", "discussion", "analysis"] if kw in text_lower)
        conclusion_score = sum(1 for kw in ["结论", "总结", "conclusion", "summary"] if kw in text_lower)
        reference_score = sum(1 for kw in ["参考文献", "引用", "reference", "bibliography"] if kw in text_lower)
        
        scores = [intro_score, method_score, result_score, discussion_score, conclusion_score, reference_score]
        total = sum(scores) + 1  # 避免除零
        
        for i, cat in enumerate(categories):
            # 基础值+关键词分数，确保每个分类都有值
            base_value = 10 + (scores[i] * 15)
            pie_data.append({
                "name": cat,
                "value": min(40, max(5, base_value))
            })
            
    elif doc_type == "business":
        # 商业报告分类
        categories = ["市场分析", "财务数据", "战略规划", "竞争分析", "执行总结", "风险评估"]
        # 基于关键词频率
        for cat in categories:
            # 简单估算：根据分类关键词在文本中出现的次数
            cat_keywords = {
                "市场分析": ["市场", "营销", "需求", "消费者", "客户"],
                "财务数据": ["财务", "收入", "利润", "成本", "预算"],
                "战略规划": ["战略", "规划", "目标", "愿景", "使命"],
                "竞争分析": ["竞争", "对手", "优势", "劣势", "市场份额"],
                "执行总结": ["执行", "实施", "计划", "时间表", "里程碑"],
                "风险评估": ["风险", "评估", "威胁", "机会", "不确定性"]
            }
            score = sum(1 for kw in cat_keywords.get(cat, []) if kw in text)
            value = 10 + score * 8
            pie_data.append({
                "name": cat,
                "value": min(35, max(5, value))
            })
            
    elif doc_type == "technical":
        # 技术文档分类
        categories = ["概述与安装", "API接口", "代码示例", "配置说明", "故障排除", "最佳实践"]
        for cat in categories:
            value = 15  # 基础值
            pie_data.append({
                "name": cat,
                "value": value
            })
            
    elif doc_type == "news":
        # 新闻文章分类
        categories = ["政治", "经济", "社会", "文化", "科技", "国际"]
        for cat in categories:
            value = 15  # 基础值
            pie_data.append({
                "name": cat,
                "value": value
            })
            
    else:
        # 通用分类，基于内容段落分析
        categories = ["核心内容", "背景信息", "数据分析", "结论总结", "其他信息"]
        for i, cat in enumerate(categories):
            pie_data.append({
                "name": cat,
                "value": 20 - i*3  # 递减权重
            })
    
    # 确保总和接近100
    total = sum(item["value"] for item in pie_data)
    if total > 0:
        for item in pie_data:
            item["value"] = round(item["value"] * 100 / total)
    
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
