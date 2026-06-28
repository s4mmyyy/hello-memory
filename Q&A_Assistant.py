"""
智能文档问答助手 - 基于HelloAgents的智能文档问答系统

这是一个完整的PDF学习助手应用，支持：
- 加载PDF文档并构建知识库
- 智能问答（基于RAG）
- 学习历程记录（基于Memory）
- 学习回顾和报告生成
"""

from dotenv import load_dotenv
load_dotenv()
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from hello_agents.tools import MemoryTool, RAGTool
import gradio as gr

class PDFlearningAssistant:
    """智能文档问答助手"""

    def __init__(self, user_id: str = "default_user"):
        """初始化学习助手

        Args:
            user_id: 用户ID，用于隔离不同用户的数据
        """
        self.user_id = user_id
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}" #用时间戳代替session_id

        # 初始化工具
        self.memory_tool = MemoryTool(user_id=user_id) 
        # 通过user_id参数实现用户级别的记忆隔离。不同用户的学习记忆是完全独立的，每个用户都有自己的工作记忆、情景记忆、语义记忆和感知记忆空间。
        
        self.rag_tool = RAGTool(rag_namespace=f"pdf_{user_id}")
        # 通过rag_namespace参数实现知识库的命名空间隔离。使用f"pdf_{user_id}"作为命名空间，每个用户都有自己独立的PDF知识库。

        # 学习统计
        self.stats = {
            "session_start": datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
            "concepts_learned": 0,
        } #stats字典记录关键的学习指标，用于生成学习报告

        # 当前加载的文档
        self.current_document = None
    
    def load_document(self, pdf_path: str) -> Dict[str, Any]:
        """加载PDF文档到知识库

        Args:
            pdf_path: PDF文件路径

        Returns:
            Dict: 包含success和message的结果
        """
        if not os.path.exists(pdf_path):
            return {"success":False, "message": f"文件不存在：{pdf_path}"}
        
        start_time = time.time()

        try:
        # 【RAGTool】处理PDF：MarkItDown转换 -> 智能分块 -> 向量化
            result = self.rag_tool.run({
                "action":"add_document",
                "file_path":pdf_path,
                "chunk_size":1000,
                "chunk_overlap":200
            })

            process_time = time.time() - start_time

            #RAG工具返回的是字符串消息
            self.current_document = os.path.basename(pdf_path)
            self.stats["documents_loaded"] += 1