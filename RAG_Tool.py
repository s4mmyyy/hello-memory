
from typing import Dict, Any, List, Optional
import os
import time
from package.base import Tool, ToolParameter, tool_action
from package.memory.rag.pipeline import create_rag_pipeline
from package.core.llm import HelloAgentsLLM
from dotenv import load_dotenv
load_dotenv()

class RAGtOOL(Tool):
    """RAG工具

    提供完整的 RAG 能力：
    - 添加多格式文档（PDF、Office、图片、音频等）
    - 智能检索与召回
    - LLM 增强问答
    - 知识库管理
    
    """

    def __init__(
            self,
            knowledge_base_path: str = "./knowledge_base",
            qdrant_url: str = None,
            qdrant_api_key: str = None,
            collection_name: str = "rag_knoledge_base",
            rag_namespace: str = "default"
            
    ):
        # 初始化RAG管道
        self._pipelines: Dict[str, Dict[str, Any]] = {}
        self.llm = HelloAgentsLLM()

        # 创建默认管道

        default_pipeline = create_rag_pipeline(
            qdrant_url=self.qdrant_url,
            qdrant_api_key=self.qdrant_api_key,
            collection_name=self.collection_name,
            rag_namespace=self.rag_namespace
        )

        self._pipelines[self.rag_namespace] = default_pipeline