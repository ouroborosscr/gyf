"""
frp_kb.py — FRP 知识库加载 + 向量化

设计：
    1. 启动时把 knowledge_base/frp/*.json 全部读进内存
    2. 对每个文档生成「描述向量」：用 embeddings.embed_query 把 
       matching_features.text_description + 拼接的 samples flow_summary 编码
    3. 给定 target_vector（流量批次的向量），用 cosine 算与各文档的相似度
    4. 返回 top-K 文档

注意：
    - 文档向量是「文本语义」向量（来自描述），target 是「行为+payload 文本」向量
    - 严格按照 evaluate_recall_ip_port.py 的方式，target 向量长度 = 38 维统计 + payload embedding 维度
    - 文档向量这边只用 payload embedding 部分（无行为统计）做 cosine
    - 所以匹配的是「流量的 payload 文本描述」与「文档的特征文本描述」的语义相似度
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


logger = logging.getLogger(__name__)


class FRPKnowledgeBase:
    """FRP 知识库的内存表示 + 向量化"""
    
    def __init__(self, kb_dir: str, embed_fn=None):
        """
        Args:
            kb_dir: knowledge_base/frp 目录路径
            embed_fn: 文本→向量的函数。签名 fn(text: str) -> np.ndarray
                      默认 None 时用 llm.embeddings.embed_query
        """
        self.kb_dir = Path(kb_dir)
        self.embed_fn = embed_fn or self._default_embed
        
        self.docs: List[Dict[str, Any]] = []          # 顶层工具文档（目前只有 frp）
        self.skills: Dict[str, Dict[str, Any]] = {}   # skill_id → skill content
        self._loaded = False
    
    def _default_embed(self, text: str) -> np.ndarray:
        """默认 embedder：从 gyf 项目的 llm.py 拿 embeddings"""
        try:
            from llm import embeddings
            vec = embeddings.embed_query(text)
            arr = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            return arr / norm if norm > 0 else arr
        except Exception as e:
            logger.warning(f"默认 embed 失败 ({e})，使用零向量")
            return np.zeros(2560, dtype=np.float32)
    
    def load(self):
        """加载所有 JSON + 向量化"""
        if self._loaded:
            return
        
        # 1. 加载顶层文档
        for top_json in self.kb_dir.glob("*.json"):
            with open(top_json, "r", encoding="utf-8") as f:
                doc = json.load(f)
            
            # 拼接用于 embedding 的文本：description + samples
            mf = doc.get("matching_features", {})
            text_chunks = [mf.get("text_description", "")]
            for s in mf.get("samples", []):
                text_chunks.append(s.get("flow_summary", ""))
            combined_text = "\n".join(c for c in text_chunks if c)
            
            doc["_embedding"] = self.embed_fn(combined_text)
            doc["_doc_text"] = combined_text
            
            self.docs.append(doc)
            logger.info(f"  加载文档: {doc.get('doc_id')} (text={len(combined_text)} chars)")
        
        # 2. 加载所有 skills
        skills_dir = self.kb_dir / "skills"
        if skills_dir.is_dir():
            for skill_json in skills_dir.glob("*.json"):
                with open(skill_json, "r", encoding="utf-8") as f:
                    skill = json.load(f)
                
                mf = skill.get("matching_features", {})
                text_chunks = [mf.get("text_description", "")]
                for s in mf.get("samples", []):
                    text_chunks.append(s.get("flow_summary", ""))
                combined_text = "\n".join(c for c in text_chunks if c)
                
                skill["_embedding"] = self.embed_fn(combined_text)
                skill["_skill_text"] = combined_text
                
                self.skills[skill["skill_id"]] = skill
                logger.info(f"  加载 skill: {skill.get('skill_id')}")
        
        self._loaded = True
        logger.info(f"知识库加载完成: {len(self.docs)} 文档, {len(self.skills)} skills")
    
    def get_skill(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """通过 skill_id 取出 skill 内容"""
        return self.skills.get(skill_id)
    
    def recall_docs(
        self,
        target_vector: np.ndarray,
        top_k: int = 3,
        only_payload_dim: bool = True
    ) -> List[Dict[str, Any]]:
        """
        根据 target_vector 召回 top-K 文档。
        
        target_vector 是 evaluate_recall_ip_port.py 风格的融合向量：
        [38 维行为统计 || payload embedding]
        
        Args:
            target_vector: 待测流量的特征向量
            top_k: 召回数量
            only_payload_dim: 只用 payload embedding 部分做匹配（默认 True）
        
        Returns:
            [{doc_id, score, doc_content}, ...] 降序
        """
        if not self._loaded:
            self.load()
        
        results = []
        for doc in self.docs:
            doc_emb = doc["_embedding"]  # 文档侧只有 payload embedding 维度
            
            if only_payload_dim and len(target_vector) > len(doc_emb):
                # 切出 target 的 payload 部分（跳过 38 维统计）
                target_payload = target_vector[-len(doc_emb):]
            else:
                target_payload = target_vector
            
            # 维度对齐
            min_dim = min(len(target_payload), len(doc_emb))
            t = target_payload[:min_dim]
            d = doc_emb[:min_dim]
            
            tn = np.linalg.norm(t)
            dn = np.linalg.norm(d)
            score = float(np.dot(t, d) / (tn * dn)) if tn > 0 and dn > 0 else 0.0
            
            results.append({
                "doc_id": doc.get("doc_id"),
                "score": score,
                "doc_content": doc
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


def build_target_vector(flows: List[Dict[str, Any]], embed_fn=None) -> np.ndarray:
    """
    用 rag.py 里的函数把一批流转成融合向量。
    
    完全复用 _build_batch_profile 和 _extract_payload_summary，
    保证和 evaluate_recall_ip_port.py 的召回口径一致。
    """
    # 复用 rag.py
    try:
        from rag import _build_batch_profile, _extract_payload_summary
    except ImportError as e:
        raise ImportError(
            "无法导入 rag._build_batch_profile/_extract_payload_summary，"
            f"请确保在 gyf 项目根目录运行测试: {e}"
        )
    
    if embed_fn is None:
        from llm import embeddings
        def embed_fn(t):
            return np.array(embeddings.embed_query(t), dtype=np.float32)
    
    if not flows:
        # 默认 38 + 2560
        return np.zeros(38 + 2560, dtype=np.float32)
    
    # 1. 38 维行为统计
    stats_vec = _build_batch_profile(flows)
    
    # 2. payload embedding
    payload_text = _extract_payload_summary(flows)
    if payload_text:
        payload_vec = embed_fn(payload_text)
        n = np.linalg.norm(payload_vec)
        if n > 0:
            payload_vec = payload_vec / n
    else:
        # 占位，长度未知时用默认 2560
        payload_vec = np.zeros(2560, dtype=np.float32)
    
    # 3. 加权拼接 (和 evaluate_recall_ip_port.py 一致)
    WEIGHT_STATS = 0.6
    WEIGHT_PAYLOAD = 0.4
    combined = np.concatenate([
        np.sqrt(WEIGHT_STATS) * stats_vec,
        np.sqrt(WEIGHT_PAYLOAD) * payload_vec
    ])
    norm = np.linalg.norm(combined)
    return combined / norm if norm > 0 else combined
