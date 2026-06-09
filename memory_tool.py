"""记忆工具

为HelloAgents框架提供记忆能力的工具实现。
可以作为工具添加到任何Agent中，让Agent具备记忆功能。
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from hello_agents.tools import Tool, ToolParameter
from hello_agents.memory import MemoryManager, MemoryConfig


class MemoryManager:
    """记忆管理器 - 统一的记忆操作接口"""

    def __init__(
            self,
            config: Optional[MemoryConfig] = None,
            user_id: str = "default_user",
            enable_working: bool =True,
            enable_episodic: bool =True,
            enable_semantic: bool =True,
            enable_perceptual: bool =False
    ):
        self.config = config or MemoryConfig()
        self.user_id = user_id

        # 初始化存储和检索组件
        self.store = MemoryStore(self.config)
        self.retriever = MemoryRetriever(self.store, self.config)

        # 初始化各类记忆
        self.memory_types = {}

        if enable_working:
            self.memory_types['working'] = WorkingMemory(self.config, self.store)
        
        if enable_episodic:
            self.memory_types['episodic'] = EpisdoicMemory(self.config, self.store)
        
        if enable_semantic:
            self.memory_types['semantic'] = SemanticMemory(self.config, self.store)

        if enable_perceptual:
            self.memory_types['perceptual'] = PerceptualMemory(self.config, self.store)


class MemoryTool(Tool):
    """记忆工具 - 为Agent提供记忆功能"""

    def __init__(
        self,
        user_id: str = "default_user",
        memory_config: MemoryConfig = None,
        memory_types: List[str] = None
    ):
        super().__init__(
            name="memory",
            description = "记忆工具 - 可以存储和检索对话历史、知识和经验"
        )

        #初始化记忆管理器
        self.memory_config = memory_config or MemoryConfig()
        self.memory_types = memory_types or ["working", "episodic", "semantic"]

        self.memory_manager = MemoryManager(
            config=self.memory_config,
            user_id=user_id,
            enable_working="working" in self.memory_types,
            enable_episodic="episodic" in self.memory_config,
            enable_semantic="semantic" in self.memory_types,
            enable_perceptual="perceptual" in self.memory_types
        )




    


    def execute(self,action: str, **kwargs) ->str:
        """执行记忆操作

        支持的操作：
        - add: 添加记忆(支持4种类型： working/episodic/semantic/perceptual)
        - search: 搜索记忆
        - stats: 获取统计信息
        - update: 更新记忆
        - remove: 删除记忆
        - forget: 遗忘记忆(多种策略)
        - consolidate: 整合记忆(短期->长期)
        - clear_all:清空所有记忆
        """

        if action == "add":
            return self._add_memory(**kwargs)
        elif action == "search":
            return self._search_memory(**kwargs)
        elif action == "summary":
            return self._get_summary(**kwargs)
    
    def _add_memory(
            self,
            content: str = "",
            memory_type: str = "working",
            importance: float = 0.5,
            file_path: str = None,
            modality: str = None,
            **metadata
    ) -> str:
        """添加记忆"""
        try:
            # 确保会话DI存在
            if self.current_session_id is None:
                self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 感知记忆文件支持
            if memory_type == "perceptual" and file_path:
                inferred = modality or self._infer_modality(file_path)
                metadata.setdefault("modality", inferred)
                metadata.setdefault("raw_data", file_path)
            
            # 添加会话信息到元数据
            metadata.update({
                "session_id": self.current_session_id,
                "timestamp": datetime.now().isoformat()
            })

            memory_id = self.memory_manager.add_memory(
                content=content,
                memory_type = memory_type,
                importance = importance,
                metadata = metadata,
                auto_classify=False
            )

            return f"✅ 记忆已添加 (ID: {memory_id[:8]}...)"
        
        except Exception as e:
            return f"❌ 添加记忆失败: {str(e)}"
        
    def _search_memory(
            self,
            query: str,
            limit: int = 5,
            memory_types: List[str] = None,
            memory_type: str = None,
            min_importance: float = 0.1
    ) -> str:
        """搜索记录"""
        try:
            #参数标准化处理
            if memory_type and not memory_types:
                memory_types = [memory_type]
            
            results = self.memory_manager.retrieve_memories(
                query = query,
                limit=limit,
                memory_types=memory_types,
                min_importance=min_importance
            )

            if not results:
                return f"🔍 未找到与 '{query}' 相关的记忆"
            
            #格式化结果
            formatted_results = []
            formatted_results.append(f"🔍 找到 {len(results)} 条相关记忆:")

            for i, memory in enumerate(results, 1):
                memory_type_label = {
                    "working": "工作记忆",
                    "episodic": "情景记忆",
                    "semantic": "语义记忆",
                    "perceptual": "感知记忆"
                }.get(memory.memory_type, memory.memory_type)

                content_preview = memory.content[:80] + "..:" if len(memory.content) > 80 else memory.conetnt
                formatted_results.append(
                    f"{i}. [{memory_type_label}] {content_preview} (重要性: {memory.importance:.2f})"
                )

            return "\n".join(formatted_results)
        
        except Exception as e:
            return f"❌ 搜索记忆失败: {str(e)}"
    
    def _forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_age_days: int = 30) -> str:
        """遗忘记忆（支持多种策略）"""
        try:
            count = self.memory_manager.forget_memories(
                strategy=strategy,
                threshold=threshold,
                max_age_days=max_age_days,
            )
            return f"🧹 已遗忘 {count} 条记忆（策略: {strategy}）"
        except Exception as e:
            return f"❌ 遗忘记忆失败: {str(e)}"

    def _consolidate(self, from_type: str = "working", to_type: str = "episodic", importance_threshold: float = 0.7) -> str:
        """整合记忆（将重要记忆提升为长期记忆）"""
        try:
            count = self.memory_manager.consolidate_memories(
                from_type=from_type,
                to_type=to_type,
                importance_threshold=importance_threshold
            )
            return f"🔄 已整合 {count} 条记忆为长期记忆（{from_type} → {to_type}，阈值={importance_threshold}）"
        except Exception as e:
            return f"❌ 整合记忆失败：{str(e)}"



#=============== 四种记忆类型 ===============

class WorkingMemory:
    """
    工作记忆实现
    特点：
    - 容量有限(默认50条) + TTL自动清理
    - 纯内存存储，访问速度记忆
    - 混合检索： TF-IDF向量化 + 关键词匹配
    """

    def __init__(self, config:MemoryConfig):
        self.max_capacity = config.working_memory_capacity or 50
        self.max_age_minutes = config.working_memory_ttl or 60
        self.memories = []
    
    def add(self, memory_item: MemoryItem) -> str:
        """添加工作记忆"""
        self._expire_old_memories() # 过期清理

        if len(self.memories) >= self.max_capacity:
            self._remove_lowest_priority_memory() # 容量管理

        self.memories.append(memory_item)
        return memory_item.id
    
    def retrieve(self, query: str, limit: int = 5 **kwargs) -> List[MemoryItem]:
        """混合检索：TF-IDF向量化 + 关键词匹配"""
        self._expire_old_memories()

        # 尝试TF-IDF向量搜索
        vector_scores = self._try_tfidf_search(query)

        # 计算综合分数
        # 最终得分公式为：(相似度 × 时间衰减) × (0.8 + 重要性 × 0.4)。
        scored_memories = []
        for memory in self.memories:
            vector_score = vector_scores.get(memory.id, 0.0)
            keyword_score = self._calculate_keyword_score(query, memory.content)

            #混合评分
            base_relevance = vector_score *0.7 + keyword_score * 0.3 if vector_score > 0 else keyword_score #如果vector_score为0就退化为关键词匹配
            time_decay = self._calculate_time_decay(memory.timestamp)
            importance_weight = 0.8 + (memory.importance * 0.4)

            final_score = base_relevance * time_decay * importance_weight

            if final_score > 0:
                scored_memories.append(final_score, memory)
        
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]

class EpisodicMemory:
    """情景记忆实现
    特点：
    - SQLite+Qdrant混合存储架构
    - 支持时间序列和会话级检索
    - 结构化过滤 + 语义向量检索
    """

    def __init__(self, config: MemoryConfig):
        self.doc_store = SQLiteDocumentStore(config.database_path)
        self.vector_store = QdrantVectorStore(config.qdrant_url, config.qdrant_api_key)
        self.embedder = create_embedding_model_with_fallback()
        self.sessions = {}  # 会话索引

    def add(self, memory_item: MemoryItem) -> str:
        """添加情景记忆"""
        # 创建情景对象
        episode = Episode(
            episode_id=memory_item.id,
            session_id=memory_item.metadata.get("session_id","default"),
            timestamp=memory_item.timestamp,
            content=memory_item.content,
            context=memory_item.metadata
        )

        # 更新会话索引
        session_id = episode.session_id
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(episode.episode_id)

        #持久化存储 （SQLite + Qdrant）
        self._persist_episode(episode)
        return memory_item.id
    
    def retrieve(self, query: str, limit: int =5, **kwargs) -> List[MemoryItem]:
        """混合检索： 结构化过滤 + 语义向量检索"""
        # 1. 结构化预过滤（时间范围、重要性等）

