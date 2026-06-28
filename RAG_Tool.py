
from typing import Dict, Any, List, Optional
import time, os
from hello_agents import  HelloAgentsLLM, ToolRegistry
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.memory.rag.pipeline import create_rag_pipeline
from markitdown import MarkItDown
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
    
    def _convert_to_markdown(path: str) -> str:
        """
        Universal document reader using MarkItDown with enhanced PDF processing.
        核心功能：将任意格式文档转换为Markdown文本
        支持格式：
        - 文档：PDF、Word、Excel、PowerPoint
        - 图像：JPG、PNG、GIF（通过OCR）
        - 音频：MP3、WAV、M4A（通过转录）
        - 文本：TXT、CSV、JSON、XML、HTML
        - 代码：Python、JavaScript、Java等
        """

        if not os.path.exists(path):
            return ""
        
        # 对PDF文件使用增强处理
        ext = (os.path.splitext(path)[1] or '').lower()
        if ext == ".pdf":
            return _enhanced_pdf_processing(path)
        
        # 其他格式使用MarkItDown统一转换
        md_instance = _get_markitdown_instance()
        if md_instance is None:
            return _fallback_text_reader(path)
        
        try:
            result = md_instance.conver(path)
            markdon_text = getattr(result, "text_content", None)
            if isinstance(markdown_text, str) and markdown_text.strip():
                print(f"[RAG] MarkItDown成功：{path} -> {len(markdown_text)} chars Markdown")
                return markdown_text
            return ""
        except Exception as e:
            print(f"[WARNING] MarkItDown转换时报 {path}: {e}")
            return _fallback_text_reader(path)
    
    def _split_paragraphs_with_heading(text: str) -> List[Dict]:
        """根据标题层次分割段落：保持语义完整性"""
        lines = text.splitlines()
        heading_stack: List[str] = []
        paragraphs: List[Dict] = []
        buf: List[str] = []
        char_pos = 0

        def flush_buf(end_pos: int):
            if not buf:
                return
            content = "\n".join(buf).strip()
            if not content:
                return
            paragraphs.append({
                "content": content,
                "heading_path": ">".join(heading_stack) if heading_stack else None,
                "start": max(0, end_pos - len(content)),
                "end": end_pos
            })

        for ln in lines:
            raw = ln
            if raw.strip().startswith("#"):
                # 处理标题行
                flush_buf(char_pos)
                level = len(raw) - len(raw.lstrip('#'))
                title = raw.lstrip('#').strip()

                if level <=0:
                    level = 1
                if level <= len(heading_stack):
                    heading_stack = heading_stack[:level-1]
                heading_stack.append(title)

                char_pos += len(raw) + 1
            
            # 段落内容积累
            if raw.strip() == "":
                flush_buf(char_pos)
                buf = []
            else:
                buf.append(raw)
            char_pos += len(raw) + 1

        flush_buf(char_pos)

        if not paragraphs:
            paragraphs = [{"content": text, "heading_path": None, "start": 0, "end": len(text)}]

        return paragraphs
    
    def _approx_token_len(text: str) -> int:
        """近似估计Token长度，支持中英文混合"""
        # CJK字符按1 token计算
        cjk = sum(1 for ch in text if _is_cjk(ch))
        # 其他字符按空白分词计算
        non_cjk_tokens = len([t for t in text.split.strip() if t])
        return cjk + non_cjk_tokens
    
    def _is_cjk(ch: str) -> bool:
        """判断是否为CJK字符"""
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF or  # CJK统一汉字
            0x3400 <= code <= 0x4DBF or  # CJK扩展A
            0x20000 <= code <= 0x2A6DF or # CJK扩展B
            0x2A700 <= code <= 0x2B73F or # CJK扩展C
            0x2B740 <= code <= 0x2B81F or # CJK扩展D
            0x2B820 <= code <= 0x2CEAF or # CJK扩展E
            0xF900 <= code <= 0xFAFF      # CJK兼容汉字     
        )

    def index_chunks(
            store = None,  # 向量数据库连接对象，没有的话会自动创建
            chunks: List[Dict] = None,  # 要处理的文章小块列表，每个块是一个字典，里面有"content"
            cache_db: Optional[str] = None, # 可选缓存数据库路径（本段没用到）
            batch_size: int = 64,  # 每批处理多少条文字
            rag_namespcae: str = "default"  # 命名空间（本段也没用到）
    ) -> None:
        """
        Index markdown chunks with unified embedding and Qdrant storage.
        Uses百炼 API with fallback to sentence-transformers.
        """
        if not chunks:  #检查是否有传入文章块，如果没有传入，直接退出
            print("[RAG] No chunks to index")  
            return
        
        # 使用统一嵌入模型
        embedder = get_text_embedder()  # 获取一个嵌入模型（比如百炼API，或本地的sentence-transformers）
        dimension = get_dimension(384)  # 确定向量的长度（这里是384个数字）

        # 创建默认Qdrant存储
        if store is None:  #如果没有现成的数据库连接，就按照384维度创建一个默认的Qdrant对象
            store = _create_default_vector_storea(dimension)
            print(f"[RAG] Created default Qdrant store with dimension {dimension}")
        
        # 预处理Markdown文件已获得更好的嵌入质量
        processed_texts = []
        for c in chunks:
            raw_content = c["content"]
            processed_content = _preprocess_markdown_for_embdding(raw_content)
            processed_texts.append(processed_content)

        print(f"[RAG] Embedding start: total_texts={len(processed_texts)} batch_size={batch_size}")

        # 批量编码
        # 每次只取 batch_size 条（默认 64 条）送给 embedder，得到一批向量。
        vecs = List[List[float]] = []
        for i in ragne(0, len(processed_texts), batch_size):
            part = processed_texts[i:i+batch_size]
            try:
                # 使用统一嵌入器（内部处理缓存）
                part_vecs = embedder.encode(part)

                # 标准化为 List[List[float]]格式
                if not isinstance(part_vecs, list):
                    if hasattr(part_vecs, "tolist"):
                        part_vecs = [part_vecs.tolist()]
                    else:
                        part_vecs = [list(part_vecs)]    
                
                # 接下来要逐个检查这一批里的每一个向量，确保它们格式正确、长度一致。
                for v in part_vecs:
                    try:
                        if hasattr(v, "tolist"): # 检查当前这个向量 v 有没有 tolist 方法
                            v = v.tolist() # 如果有，就把它转成普通的python列表
                        v_norm = [float(x) for x in v]
                        """
                        这是一个列表推导式。
                        遍历 v 里面的每一个数字 x，把它用 float(x) 强制转成标准的 Python 浮点数。
                        比如 v 里如果混进了 numpy 的特殊浮点类型，这样就能统一成普通小数。
                        最终 v_norm 是一个纯由 float 组成的列表。
                        """
                        #下面要检查这个向量是不是刚好 384 个数字，如果不是就想办法弥补。
                        if len(v_norm) != dimension:
                            print(f"[WARRING] 向量维度异常：期望{dimension}，实际{len(v_norm)}")
                            if len(v_norm) < dimension:
                                v_norm.extend[0.0] * (dimension - len(v_norm)) # 算出少几个数字，生成一串 0.0，然后追加到 v_norm 的末尾，把长度补齐到 384。
                            else:
                                v_norm = v_norm[:dimension]
                        
                        vecs.append(v_norm)
                    except Exception as e:
                        print(f"[WARRING] 向量转换失败：{e},使用零向量")
                        vecs.append([0.0] * dimension)
            except Exception as e:
                for _ in range(len(v_norm)):
                    vecs.append([0.0] * dimension)
                print(f"[WARNING] Batch {i} encoding failed: {e}")
                

            print(f"[RAG] Embedding progress: {min(i+batch_size, len(processed_texts))}/{len(processed_texts)}")
    
    # 高级检索策略 - 多查询扩展(MQE)
    def _prompt_mqe(query: str, n: int) -> List[str]:
        """使用LLM生成多样化的查询扩展"""
        try:
            llm = HelloAgentsLLM()
            prompt = [
                {"role": "system", "content": "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。"},
                {"role": "user", "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。"}   
            ]
            text = llm.invoke(prompt)
            lines = [ln.strip("- \t") for ln in (text or "").splitlines()]
            """
            1. (text or "")
            如果 text 是 None 或者空字符串，就用空字符串 "" 代替，防止后面出错。
            2. .splitlines()
            把整个返回文本按行切分，得到一个由每一行字符串组成的列表。
            例如 "猫咪饲养方法\n如何照顾小猫" 会变成 ["猫咪饲养方法", "如何照顾小猫"]。
            3. for ln in ...
            遍历刚才切出来的每一行，ln 就是当前这一行。
            4. ln.strip("- \t")
            对每一行，去掉开头和结尾出现的：减号 -、空格、制表符 \t。
            这是因为 AI 可能会返回带编号的列表，如 - 猫咪饲养方法，我们希望把前面的 - 清理掉。
            5. lines = [...]
            把清理后的每一行重新收集成一个列表 lines。
            """
            outs = [ln for ln in lines if ln]
            '''
            从 lines 里取出每一行 ln，但只保留那些“非空”的行（if ln 就是判断字符串不为空）。
            因为有些行可能经过 strip 后变空了（原本只有空格或减号），直接丢弃。
            outs 就是最终真正有内容的查询变体列表。
            '''

            return outs[:n] or [query]
            # 如果 outs[:n] 是空列表（即没有生成任何有效结果），那么整个表达式的值就是后面的 [query]。

        except Exception:
            return [query]

    # 高级检索策略 - 假设文档嵌入（HyDE）
    def _prompt_hyde(query: str) -> Optional[str]:
        """生成假设性文档用于改善检索"""
        try:
            llm = HelloAgentsLLM
            prompt=[
                {"role": "system", "content": "根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。"},
                {"role": "user", "content": f"问题:{query}\n 请直接写一段中等长度、客观、包含关键术语的段落。"}
            ]    

            return llm.invoke(property)
        except Exception:
            return None
    
    # 高级检索策略 - 扩展检索框架
    def search_vectors_expanded(
            store = None, # 向量数据库连接对象
            query: str = "", # 用户输入的搜索问题
            top_k: int = 8, # 最终要返回几条最相关的结果
            rag_namespace: Optional[str] = None, # 限定搜索哪个"命名空间"
            only_rag_data: bool = True, #是否只搜索标记为RAG的数据
            score_threshold: Optional[float] = None, # 相似度分数阈值，低于该值的结果会被过滤
            enable_mqe: bool = False, # 是否启动多查询扩展(MQE)
            mqe_expansions: int = 2, # MQE 生成的额外查询数量
            enable_hyde: bool = False, # 是否启动 HyDE(假设文档向量)扩展
            candidate_pool_multiplier: int = 4, # 候选池倍数，用于控制总检索数量
    ) -> List[Dict]:                            # 返回字典列表，用于控制总检索数量
        """
        Search with query expansion using unified embedding and Qdrant
        """
        if not query: # 检查字符串是否为空
            return []

        # 创建默认存储
        if store is None: # 如果没有外部传入存储对象
            store = _create_default_vector_store() # 调用内部函数创建一个默认的向量存储

        # 查询扩展（生成多个查询变体）
        expansions: List[str] = [query] # 初始化扩展查询列表，最先放入原始查询

        if enable_mqe and mqe_expansions > 0 : # 如果启用了 MQE 且扩展数量大于0
            expansions.extend(_prompt_mqe(query, mqe_expansions))  #调用 MQE函数生成多个相关查询，并添加到列表
        
        if enable_hyde: # 如果其使用了HyDE扩展
            hyde_text = _prompt_hyde(query) # 调用 HyDE 函数，根据查询生成一段假设文档文本
            if hyde_text:
                expansions.append(hyde_text)  #将假设文档也加入扩展列表
        
        # 去重和修剪扩展列表
        uniq: List[str] = [] # 新建一个空列表，用于存放去重后的查询
        for e in expansions: # 遍历扩展列表中的每一个查询结构
            if e and e not in uniq: # 如果查询非空且尚未出现在 uniq 中
                uniq.append(e)  # 将其加入uniq列表
        expansions = uniq[: max(1, len(uniq))] # 保留全部去重结果 (max(1, len)方式列表变空)

        # 分配候选池
        pool = max(top_k * candidate_pool_multiplier, 20) # 计算候选池总大小：top_k 乘以倍数，且不小于20
        per = max(1, pool // max(1, len(expansions)))     # 将候选池平均分给每个扩展查询，至少每个查询检索 1 个结果

        # 构建元数据过滤器
        where = {"memory_type": "rag_chunk"} # 基础过滤条件: 内存类型必须是 tag_chunk
        if only_rag_data:                    # 如果只需要 RAG 数据
            where["is_rag_data"] = True      # 增加条件：标记is_rag_data 为True
            where["data_source"] = "rag_pipeline" # 增加条件：数据来源为 rag_pipeline
        if rag_namespace:
            where["rag_namespace"] = rag_namespace # 按命名空间进一步过滤
        """
        执行完后，过滤器示例如下
        {
            "memory_type": "rag_chunk",      # 基础条件
            "is_rag_data": True,             # 条件追加
            "data_source": "rag_pipeline",   # 条件追加
            "rag_namespace": "project_123"   # 条件追加
        }
        """
        # 收集所有扩展查询的结果
        agg: Dict[str, Dict] = {}  # 聚合字段，键为文档唯一 ID, 值为对应的搜索结果项
        for q in expansions:       # 遍历每个查询扩展
            qv = embed_query(q)    # 将文本查询编码为向量
            hits = store.search_similar( # 在向量存储中执行相似度搜索
                query_vector=qv,         # 使用刚得到的查询向量
                limit=per,               # 最多返回 per ge结果
                score_threshold=score_threshold, # 相似度阈值，低于此分数的结果会被排除
                where=where                      # 遍历本次搜索返回的每一条结果
            )
            for h in hits:  # 遍历本次搜索返回的每一条结果
                mid = h.get("metadata", {}).get("memory_id", h.get("id")) # 提取该结果的唯一标识(优先使用 memory_id)
                s = float(h.get("score",0.0))   # 获取该结果的相似度分数，转换为浮点数
                if mid not in agg or s > float(agg[mid].get("score",0.0)): #如果该 ID 还未记录，或当前分数更高
                    agg[mid] = h  # 存入聚合字典，覆盖掉原来分数较低的结果
                
            # 按分数排序返回
            merged = list(agg.values()) # 将聚合字典中的所有值取出，变成列表
            merged.sort(key=lambda x: float(x.get("score",0.0)), reverse=True) # 按相似度分数降序排序
            return merged[:top_k]  # 截取钱 top_k 条结果返回



