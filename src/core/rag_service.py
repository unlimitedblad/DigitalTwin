"""
RAG向量数据库服务模块
使用 ChromaDB 本地向量数据库，无需独立数据库服务
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import chromadb
import dashscope
from langchain_chroma import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
import logging

logger = logging.getLogger(__name__)


class RAGService:
    """RAG向量数据库服务类（基于 ChromaDB 本地存储）"""

    def __init__(
        self,
        dashscope_api_key: str,
        collection_name: str = "wechat_embeddings",
        persist_directory: str = "./chroma_db",
        embed_model: str = "text-embedding-v4",
    ):
        """
        初始化RAG服务

        Args:
            dashscope_api_key: DashScope API密钥
            collection_name: ChromaDB 集合名称
            persist_directory: ChromaDB 持久化目录（本地文件夹路径）
        """
        # 设置DashScope API密钥
        os.environ["DASHSCOPE_API_KEY"] = dashscope_api_key
        dashscope.api_key = dashscope_api_key

        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # 初始化embedding模型
        self.embeddings = DashScopeEmbeddings(model=embed_model)

        # 使用 chromadb 官方客户端，避免依赖 LangChain 私有属性
        self._chroma_client = chromadb.PersistentClient(path=persist_directory)

        # 初始化向量数据库
        self.vectorstore: Optional[Chroma] = None
        self._connect()

    def _connect(self):
        """连接（加载）本地 ChromaDB 向量数据库"""
        try:
            logger.info("正在加载 ChromaDB 本地向量数据库 (目录: %s)...", self.persist_directory)

            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )

            # 测试连接：通过官方客户端获取集合文档数量
            count = self._chroma_client.get_or_create_collection(self.collection_name).count()
            logger.info("成功连接到 ChromaDB，当前集合中共有 %s 条记录", count)

        except Exception as e:
            logger.error("连接 ChromaDB 失败: %s", e, exc_info=True)
            self.vectorstore = None
            raise ConnectionError(f"无法连接到 ChromaDB: {e}")

    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.vectorstore is not None

    def _get_nearby_records(
        self,
        timestamp: int,
        time_window_minutes: int = 30,
        max_nearby: int = 15
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        获取时间戳相近的聊天记录

        Args:
            timestamp: 目标 Unix 时间戳（整数秒）
            time_window_minutes: 时间窗口(分钟,默认30分钟)
            max_nearby: 最多返回的相近记录数(默认15条)

        Returns:
            List of (content, metadata, similarity_score)
        """
        if not self.is_connected():
            return []

        try:
            window_seconds = time_window_minutes * 60
            start_ts = timestamp - window_seconds
            end_ts = timestamp + window_seconds

            # 使用整数时间戳做数值范围过滤，避免字典序误判
            try:
                results = self.vectorstore.get(
                    where={
                        "$and": [
                            {"chat_time": {"$gte": start_ts}},
                            {"chat_time": {"$lte": end_ts}},
                        ]
                    },
                    limit=max_nearby * 2,
                    include=["documents", "metadatas"],
                )
            except Exception:
                # 若 ChromaDB 不支持该过滤器格式，退化为有限量采样，不做全量扫描
                logger.warning("时间范围过滤不支持，跳过 nearby 扩展")
                return []

            documents = results.get("documents", []) or []
            metadatas = results.get("metadatas", []) or []

            nearby_records = []
            for content, metadata in zip(documents, metadatas):
                if metadata and 'chat_time' in metadata:
                    try:
                        chat_ts = int(metadata['chat_time'])
                        if start_ts <= chat_ts <= end_ts:
                            time_diff = abs(timestamp - chat_ts)
                            score = 1.0 - (time_diff / window_seconds)
                            nearby_records.append((
                                content,
                                metadata,
                                max(0.0, score)
                            ))
                    except Exception:
                        continue

            # 按时间顺序排序并限制数量
            nearby_records.sort(key=lambda x: x[1].get('chat_time', 0))
            return nearby_records[:max_nearby]

        except Exception as e:
            logger.warning("获取相近记录失败: %s", e)
            return []

    def search(
        self,
        query: str,
        k: int = 15,
        similarity_threshold: float = 0.0,
        include_nearby: bool = True,
        time_window_minutes: int = 30,
        nearby_per_result: int = 8,
        max_total_results: int = 50,
        lambda_mult: float = 0.6,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        搜索相关聊天记录，并包含时间戳相近的记录

        Args:
            query: 查询文本
            k: MMR 最终返回结果数量(默认15)
            similarity_threshold: 保留参数，MMR 模式下不生效
            include_nearby: 是否包含时间相近的记录
            time_window_minutes: 时间窗口(分钟,默认30分钟)
            nearby_per_result: 每个结果附近获取的记录数(默认8条)
            max_total_results: 最大返回记录数(默认50,上限50)
            lambda_mult: MMR 多样性权重，0=最多样 1=最相关(默认0.6)

        Returns:
            List of (content, metadata, similarity_score)
        """
        if not self.is_connected():
            raise RuntimeError("向量数据库未连接")

        if not query.strip():
            return []

        try:
            # 用 MMR 搜索替代纯相似度搜索，在相关性和多样性之间取平衡
            # fetch_k 先召回更多候选，再从中挑选差异最大的 k 条
            fetch_k = max(k * 4, 60)
            mmr_docs = self.vectorstore.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            )

            formatted_results = []
            seen_ids = set()

            for doc in mmr_docs:
                doc_id = doc.metadata.get('id', doc.page_content[:50])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    formatted_results.append((
                        doc.page_content,
                        {**doc.metadata, '_result_source': 'semantic'},
                        1.0
                    ))

                # 如果启用了 nearby 功能，获取时间相近的记录
                if include_nearby and 'chat_time' in doc.metadata:
                    chat_ts = doc.metadata['chat_time']
                    if isinstance(chat_ts, (int, float)) and chat_ts > 0:
                        nearby_records = self._get_nearby_records(
                            timestamp=int(chat_ts),
                            time_window_minutes=time_window_minutes,
                            max_nearby=nearby_per_result
                        )
                        for content, metadata, nearby_score in nearby_records:
                            nearby_id = metadata.get('id', content[:50])
                            if nearby_id not in seen_ids:
                                seen_ids.add(nearby_id)
                                formatted_results.append((
                                    content,
                                    {**metadata, '_result_source': 'temporal'},
                                    nearby_score * 0.5
                                ))

            # 按相似度分数排序并限制总数量
            formatted_results.sort(key=lambda x: x[2], reverse=True)
            result = formatted_results[:min(max_total_results, 50)]
            logger.debug("RAG搜索完成: query='%s...', 返回 %d 条结果", query[:50], len(result))
            return result

        except Exception as e:
            logger.warning("RAG搜索异常: %s", e, exc_info=True)
            return []

    def format_context(
        self,
        results: List[Tuple[str, Dict[str, Any], float]],
        max_context_length: int = 2000,
        include_metadata: bool = True
    ) -> str:
        """
        格式化检索结果为上下文字符串

        Args:
            results: 搜索结果列表
            max_context_length: 最大上下文长度
            include_metadata: 是否包含元数据（发送者、时间等）

        Returns:
            格式化的上下文字符串
        """
        if not results:
            return ""

        lines = []
        total_length = 0

        for content, metadata, score in results:
            if include_metadata:
                chat_time = metadata.get('chat_time_str') or metadata.get('chat_time', '')
                time_prefix = f"[{chat_time}] " if chat_time else ""
                record = f"{time_prefix}{content.strip()}"
            else:
                record = content.strip()

            if total_length + len(record) > max_context_length:
                break

            lines.append(record)
            total_length += len(record)

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        if not self.is_connected():
            return {"error": "向量数据库未连接"}

        try:
            count = self._chroma_client.get_or_create_collection(self.collection_name).count()

            # 采样统计发送者/消息类型（数据量大时为近似值）
            sample_size = min(200, count)
            sample_results = self.vectorstore.get(
                limit=sample_size,
                include=["metadatas"],
            )
            metadatas = sample_results.get("metadatas", []) or []

            senders = set()
            msg_types = set()
            for metadata in metadatas:
                if metadata:
                    if 'sender' in metadata:
                        senders.add(metadata['sender'])
                    if 'msg_type' in metadata:
                        msg_types.add(metadata['msg_type'])

            return {
                "connected": True,
                "total_records": count,
                "sample_size": len(metadatas),
                "unique_senders": list(senders),
                "message_types": list(msg_types),
                "is_approximate": count > sample_size,  # 采样未覆盖全量时标注近似
                "database_host": "local",
                "database_name": f"ChromaDB ({self.persist_directory})",
            }

        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }
