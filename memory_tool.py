"""记忆工具

为HelloAgents框架提供记忆能力的工具实现。
可以作为工具添加到任何Agent中，让Agent具备记忆功能。
"""

from typing import Dict, Any, List
from datetime import datetime

from hello_agents.tools import Tool, ToolParameter
from hello_agents.memory import MemoryManager, MemoryConfig

class MemoryTool(Tool):

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
        