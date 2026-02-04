#!/usr/bin/env python3
"""
GROBID解析器模块 - 解析GROBID返回的TEI XML
"""

import io
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
import warnings

try:
    from grobid_client import Client
    from grobid_client.models import ProcessForm
    from grobid_client.api.pdf import process_fulltext_document
    from grobid_client.types import File
except ImportError:
    print("警告: grobid_client未安装")


def parse_grobid_xml(xml_content: str) -> Dict[str, Any]:
    """
    解析GROBID返回的TEI XML
    
    返回结构:
    {
        "title": str,
        "authors": List[str],
        "abstract": str,
        "sections": List[Dict],
        "references": List[Dict],
        "full_text": str,
        "structured_data": Dict  # 原始XML解析的更多字段
    }
    """
    try:
        # 注册命名空间
        ns = {
            'tei': 'http://www.tei-c.org/ns/1.0',
            'xml': 'http://www.w3.org/XML/1998/namespace'
        }
        
        root = ET.fromstring(xml_content)
        
        # 提取标题
        title = ""
        title_elem = root.find('.//tei:titleStmt/tei:title', ns)
        if title_elem is not None:
            title = ''.join(title_elem.itertext()).strip()
        
        # 提取作者
        authors = []
        for author_elem in root.findall('.//tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author', ns):
            pers_name = author_elem.find('tei:persName', ns)
            if pers_name is not None:
                forename = pers_name.find('tei:forename', ns)
                surname = pers_name.find('tei:surname', ns)
                if forename is not None and surname is not None:
                    authors.append(f"{''.join(forename.itertext()).strip()} {''.join(surname.itertext()).strip()}")
                elif surname is not None:
                    authors.append(''.join(surname.itertext()).strip())
        
        # 提取摘要
        abstract = ""
        abstract_elem = root.find('.//tei:profileDesc/tei:abstract', ns)
        if abstract_elem is not None:
            abstract = ''.join(abstract_elem.itertext()).strip()
        
        # 提取章节
        sections = []
        body_elem = root.find('.//tei:text/tei:body', ns)
        if body_elem is not None:
            for i, div in enumerate(body_elem.findall('tei:div', ns)):
                head_elem = div.find('tei:head', ns)
                section_text = ''.join(div.itertext()).strip()
                
                if head_elem is not None:
                    section_title = ''.join(head_elem.itertext()).strip()
                    # 移除标题部分的文本
                    section_content = section_text.replace(section_title, '').strip()
                else:
                    section_title = f"Section {i+1}"
                    section_content = section_text
                
                sections.append({
                    "title": section_title,
                    "content": section_content,
                    "level": div.get('type', 'section')
                })
        
        # 提取参考文献
        references = []
        for bibl in root.findall('.//tei:text/tei:back/tei:listBibl/tei:biblStruct', ns):
            title_elem = bibl.find('.//tei:title', ns)
            author_elems = bibl.findall('.//tei:author', ns)
            
            ref_title = ''.join(title_elem.itertext()).strip() if title_elem is not None else ""
            ref_authors = []
            
            for author in author_elems:
                pers_name = author.find('tei:persName', ns)
                if pers_name is not None:
                    surname = pers_name.find('tei:surname', ns)
                    if surname is not None:
                        ref_authors.append(''.join(surname.itertext()).strip())
            
            references.append({
                "title": ref_title,
                "authors": ref_authors
            })
        
        # 提取完整文本
        full_text = ""
        text_elem = root.find('.//tei:text', ns)
        if text_elem is not None:
            full_text = ''.join(text_elem.itertext()).strip()
        
        return {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "sections": sections,
            "references": references,
            "full_text": full_text,
            "structured_data": {
                "raw_xml": xml_content[:5000]  # 保存部分原始XML用于调试
            }
        }
        
    except Exception as e:
        warnings.warn(f"GROBID XML解析失败: {e}")
        return {
            "title": "",
            "authors": [],
            "abstract": "",
            "sections": [],
            "references": [],
            "full_text": "",
            "structured_data": {}
        }


def extract_with_grobid(file_bytes: bytes, grobid_url: str = "http://localhost:8070") -> Dict[str, Any]:
    """
    使用GROBID服务提取文档结构化信息
    
    Args:
        file_bytes: 文件字节
        grobid_url: GROBID服务URL
        
    Returns:
        解析后的结构化数据，如果失败返回空字典
    """
    try:
        # 创建GROBID客户端
        client = Client(base_url=grobid_url, timeout=60.0)
        
        # 准备文件
        file_obj = File(payload=io.BytesIO(file_bytes))
        
        # 创建表单数据
        multipart_data = ProcessForm(
            input_=file_obj,
            consolidate_header="1",
            consolidate_citations="1",
            segment_sentences="1"
        )
        
        # 调用GROBID API
        response = process_fulltext_document.sync_detailed(
            client=client,
            multipart_data=multipart_data
        )
        
        if response.status_code == 200:
            # 解析TEI XML
            xml_content = response.content.decode('utf-8')
            return parse_grobid_xml(xml_content)
        else:
            warnings.warn(f"GROBID API调用失败: {response.status_code}")
            return {}
            
    except Exception as e:
        warnings.warn(f"GROBID提取失败: {e}")
        return {}


def grobid_to_text(structured_data: Dict[str, Any]) -> str:
    """
    将GROBID结构化数据转换为纯文本
    
    格式:
    标题: {title}
    
    作者: {authors}
    
    摘要: {abstract}
    
    章节:
    1. {section1_title}
    {section1_content}
    
    2. {section2_title}
    {section2_content}
    ...
    """
    if not structured_data:
        return ""
    
    parts = []
    
    if structured_data.get("title"):
        parts.append(f"标题: {structured_data['title']}\n")
    
    if structured_data.get("authors"):
        authors_str = ", ".join(structured_data['authors'])
        parts.append(f"作者: {authors_str}\n")
    
    if structured_data.get("abstract"):
        parts.append(f"摘要: {structured_data['abstract']}\n")
    
    if structured_data.get("sections"):
        parts.append("章节:")
        for i, section in enumerate(structured_data['sections']):
            parts.append(f"{i+1}. {section.get('title', '')}")
            if section.get('content'):
                parts.append(f"{section['content']}")
            parts.append("")  # 空行
    
    if structured_data.get("full_text"):
        # 如果没有提取到结构化部分，使用完整文本
        if len(parts) < 3:  # 标题、作者、摘要都没有
            parts.append(structured_data['full_text'])
    
    return "\n".join(parts)


if __name__ == "__main__":
    # 测试代码
    print("GROBID解析器模块加载成功")
    print("主要函数:")
    print("  - parse_grobid_xml(xml_content): 解析GROBID XML")
    print("  - extract_with_grobid(file_bytes, grobid_url): 调用GROBID API")
    print("  - grobid_to_text(structured_data): 转换为纯文本")