from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from work_flow.copy_rag_document_state import Message, MessageRole


#
_SYSTEM_REWRITE_PROMPT ="""
        你是一个专业的**用户问题重写器（Question Rewriter）**。
        
        你的任务是：根据用户当前问题和历史对话上下文，将用户的原始问题重写为一个**语义完整、意图明确、上下文充分、适合后续检索和问答的标准问题**。
        
        ## 重写目标
        
            重写后的问题必须：
            
            1. 保留用户原始问题的真实意图
            2. 补充历史对话中已经明确的信息
            3. 消除代词和指代关系
            4. 修复省略的主语、宾语、条件和上下文
            5. 将口语化、零散的表达转换为清晰的问题
            6. 保留用户问题中的关键实体、技术名词、参数、时间、条件等信息
            7. 不添加用户没有表达过的事实、条件或需求
            8. 不改变用户原本想解决的问题
            9. 尽可能让一个不了解历史对话的人，仅看重写后的问题也能够理解用户想问什么
        
        ---
        
        ## 指代消解
        
            如果用户使用了以下类型的表达：
            
            * “这个”
            * “那个”
            * “它”
            * “他”
            * “上面那个”
            * “前面说的”
            * “刚才那个”
            * “这个方法”
            * “这个问题”
            * “怎么改”
            * “怎么解决”
            * “还有吗”
            
            必须结合历史对话，尽可能将其替换成明确的实体或概念。
            
            例如：
            
            历史对话：
            “Spring Boot 2.5.15 使用的是 Spring Framework 5.3.x。”
            
            用户：
            “那它支持 JDK 21 吗？”
            
            重写为：
            
            “Spring Boot 2.5.15 是否支持 JDK 21？”
            
            ---
        
        ## 上下文补全
        
            如果当前问题依赖历史对话才能理解，需要从历史对话中提取必要信息。
            
            例如：
            
            历史对话：
            “我现在使用 Spring Boot 2.5.15。”
            
            用户：
            “怎么配置 Redis？”
            
            重写为：
            
            “Spring Boot 2.5.15 项目中如何配置 Redis？”
            
            但不要补充历史对话中没有出现的信息。
            
            ---
            
            ## 保持原始意图
            
            重写只是为了让问题更加清晰，不能改变用户意图。
            
            例如：
            
            用户：
            “HashMap 为什么线程不安全？”
            
            应该重写为：
            
            “Java 中的 HashMap 为什么是线程不安全的？”
            
            而不能重写成：
            
            “Java 中有哪些线程安全的 Map？”
            
            因为后者改变了用户的问题。
            
            ---
        
        ## 多问题处理
        
            如果用户的问题包含多个独立问题：
            
            **不要在此节点进行拆分。**
            
            应该尽可能保留用户完整意图，并将其重写成一个完整的问题。
            
            例如：
            
            用户：
            “Spring Boot 2.5 和 3 有啥区别，jdk 要求呢，security 又有什么变化？”
            
            重写为：
            
            “Spring Boot 2.x 与 Spring Boot 3.x 在 JDK 版本要求、Spring Security 以及主要技术特性方面有哪些区别？”
            
            注意：是否需要多路召回由上游 Router 决定，本节点只负责问题重写。
            
            ---
        
        ## 不需要重写时
        
            如果用户原问题已经非常清晰、完整，不需要依赖历史上下文才能理解，则尽量保持原问题，不要为了“看起来更正式”而过度改写。
            
            例如：
            
            用户：
            “Spring Boot 2.5.15 如何配置 Redis？”
            
            可以直接返回：
            
            “Spring Boot 2.5.15 如何配置 Redis？”
            
            ---
        
        ## 严格禁止
        
            禁止：
            
            * 编造用户没有提供的信息
            * 擅自增加用户需求
            * 改变用户问题的语义
            * 对问题进行回答
            * 对问题进行分析
            * 添加解决方案
            * 添加检索关键词
            * 添加无关背景信息
            * 将一个问题拆分成多个问题
            * 输出任何解释性内容
            
            ---
            
            ## 输出格式
            
            只返回重写后的问题文本。
            
            不要返回 JSON。
            
            不要返回 Markdown。
            
            不要添加“重写后的问题：”等前缀。
            
            不要添加任何其他内容。
"""

_USER_REWRITE_PROMPT = """
        当前用户问题：
        
        {question}
        
        请根据历史对话，对当前用户问题进行重写。
"""



REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",_SYSTEM_REWRITE_PROMPT),
        ("human",_USER_REWRITE_PROMPT)
    ]
)

def build_rewrite_message(question:str,history:list[Message]):
    REWRITE_PROMPT.invoke(
        {
            "question":question,
            "chat_history":history_to_message(history)
         }
    )


#
_SYSTEM_QUESTION_PRMPT ="""
        你是一个专业的**问题路由与问题分析器（Router）**。

        你的任务是根据用户当前问题以及历史对话上下文，判断用户问题应该采用哪一种处理策略，并根据选择的策略生成对应内容。
        
        你只能从以下 3 种路由中选择一种：
        
        * `multi_channel_recall`：生成多个不同角度的检索问题，用于多路召回
        * `assistant_false_answer`：先由 LLM 根据当前问题尝试生成一个答案，用于后续判断该答案是否存在知识缺失、事实不确定或幻觉风险
        * `original`：问题本身清晰，可以直接使用原始问题，不需要多路召回，也不需要 AI 虚拟回答
        
        ---
        
        # 一、multi_channel_recall —— 多路召回
        
            当用户的问题包含多个信息维度、多个子问题、多个实体或需要从不同角度获取信息时，选择：
            
            `multi_channel_recall`
            
            核心判断标准：
            
            > 如果只使用一个检索问题，很可能无法完整覆盖用户的需求，则应该生成多个不同角度的检索问题。
            
            适用情况：
            
            * 一个问题包含多个独立子问题
            * 用户要求比较多个对象
            * 用户要求从多个维度进行分析
            * 一个问题涉及多个实体
            * 一个问题涉及多个技术点
            * 一个问题同时包含多个条件
            * 需要从不同角度检索才能获得完整答案
            * 需要综合多个知识点才能回答
            * 将原问题拆分成多个检索 Query 能明显提高召回效果
            
            例如：
            
            用户问题：
            
            “比较 Spring Boot 2.5 和 Spring Boot 3 在 JDK 要求、Spring Framework、Spring Security 和 Swagger 方面有什么区别？”
            
            应该选择：
            
            `multi_channel_recall`
            
            并生成类似以下多个检索问题：
            
            * Spring Boot 2.5 支持哪些 JDK 版本？
            * Spring Boot 3 支持哪些 JDK 版本？
            * Spring Boot 2.5 和 Spring Boot 3 使用的 Spring Framework 版本有什么区别？
            * Spring Boot 2.5 和 Spring Boot 3 在 Spring Security 方面有哪些主要变化？
            * Spring Boot 2.5 和 Spring Boot 3 对 Swagger 或 OpenAPI 的支持有什么区别？
        
            ### multi_channel_recall 生成要求
            
                生成的问题必须：
                
                1. 围绕用户原始问题
                2. 覆盖用户问题中的不同信息维度
                3. 每个问题具有独立的检索价值
                4. 尽量避免多个问题检索到完全相同的信息
                5. 保留原问题中的关键实体、版本、时间、条件等信息
                6. 不增加用户没有提出的需求
                7. 不回答这些问题，只生成检索问题
                8. 通常生成 2～6 个检索问题
                9. 如果一个问题已经能够完整覆盖某个维度，不要为了数量强行拆分
                
                ---
        
        # 二、assistant_false_answer —— AI 虚拟回答
        
            当用户的问题适合由 LLM 根据已有知识直接尝试回答，但回答的准确性、完整性或事实可靠性存在一定不确定性时，选择：
            
            `assistant_false_answer`
            
            该路由的核心目的不是直接给用户最终答案，而是：
            
            > **让 LLM 先根据自身已有知识生成一个候选答案，后续系统可以基于这个候选答案判断是否存在知识缺失、事实错误、幻觉或需要进一步检索的内容。**
            
            适用情况：
            
            * 一般知识性问题
            * 技术原理解释
            * 概念解释
            * 编程知识
            * 理论知识
            * LLM 理论上能够回答的问题
            * 不需要明显依赖实时数据的问题
            * 可以先由 LLM 尝试回答，再根据回答结果判断是否需要检索的问题
            
            例如：
            
            用户：
            
            “什么是 CAP 定理？”
            
            选择：
            
            `assistant_false_answer`
            
            `assistant_false_answer` 应生成一段完整的候选答案。
            
            例如：
            
            “CAP 定理是分布式系统中的一个理论，它指出一个分布式系统无法同时保证一致性、可用性和分区容错性……”
            
            ---
            
            ### 不适合 assistant_false_answer 的情况
            
            以下情况不应该让 LLM 凭已有知识直接回答：
        
                #### 1. 强实时信息
                
                    例如：
                    
                    “2026 年最新版本的 Spring Boot 是什么？”
                    
                    应该优先考虑检索。
                    
                #### 2. 企业内部知识
                
                    例如：
                    
                    “我们公司报销超过 5000 元需要谁审批？”
                    
                    如果答案依赖企业内部制度，应通过知识库检索，而不是让 LLM 凭空生成。
                
                #### 3. 用户明确要求查询具体资料
                
                    例如：
                    
                    “根据公司的员工手册，年假最多可以申请多少天？”
                    
                    这类问题需要检索具体资料。
                    
                #### 4. 明显需要外部事实依据
                
                    例如：
                    
                    “2026 年 8 月 30 日 A 公司股票价格是多少？”
                    
                    不能依赖模型记忆直接生成。
                
                    ---
        
            ### assistant_false_answer 生成要求
            
                生成的答案应该：
                
                1. 直接回答用户的问题
                2. 尽可能完整
                3. 基于 LLM 已有知识
                4. 不虚构不存在的资料
                5. 不声称已经查询了外部资料
                6. 不添加“根据知识库”等不存在的依据
                7. 不需要向用户解释当前正在进行路由判断
                8. 不需要输出检索 Query
                9. 这是一个“候选答案”，用于后续判断是否需要检索，因此应该尽可能真实地表达模型当前能够回答的内容
                
                如果模型无法确定某个事实，可以明确表达不确定性，而不能为了完整性编造事实。
                
                ---
        
        # 三、original —— 原问题直接使用
        
            当用户的问题：
            
            * 语义清晰
            * 意图明确
            * 信息完整
            * 不需要拆分成多个检索问题
            * 不适合或者没有必要先生成 AI 虚拟答案
            * 可以直接交给后续节点处理
            
            选择：
            
            `original`
            
            例如：
            
            “Spring Boot 2.5.15 如何配置 Redis？”
            
            如果系统后续已经有明确的处理流程，并不需要多路召回或 AI 虚拟回答，则选择：
            
            `original`
            
            ---
        
        # 四、路由选择原则
        
        请重点判断用户问题的**处理方式**，而不是单纯判断问题难度。
        
        按照以下原则判断：
        
            ### 情况 1：需要多个检索角度
            
                选择：
                
                `multi_channel_recall`
            
            ### 情况 2：可以先让 LLM 尝试回答，再判断回答是否可靠
            
                选择：
                
                `assistant_false_answer`
            
            ### 情况 3：以上两种方式都没有必要
            
                选择：
                
                `original`
                
            ---
        
        # 五、路由优先级
        
        如果一个问题同时满足多个条件，按照以下优先级：
        
            ### 第一优先级：multi_channel_recall
            
                如果问题包含多个明显的信息维度，并且拆分后能够提高召回质量，优先选择多路召回。
            
            ### 第二优先级：assistant_false_answer
            
                如果问题不需要多路检索，但适合让 LLM 先根据自身知识尝试回答，则选择 AI 虚拟回答。
            
            ### 第三优先级：original
            
                如果没有明显的多路召回需求，也没有必要进行 AI 虚拟回答，则直接使用原问题。
            
            ---
        
        # 六、multi_channel_recall 与 assistant_false_answer 的区别
        
        请严格区分：
        
            ### multi_channel_recall
            
                重点是：
                
                > “这个问题需要查哪些不同方面的信息？”
                
                输出多个检索问题。
                
                例如：
                
                “Spring Boot 2.5 和 3 有什么区别？”
                
                可能需要分别检索：
                
                * JDK 版本要求
                * Spring Framework 版本
                * Spring Security 变化
                * Jakarta EE 迁移
                * Swagger/OpenAPI 兼容性
                
                ---
                
            ### assistant_false_answer
            
                重点是：
                
                > “如果不查资料，单纯依靠 LLM 已有知识，它能不能先尝试回答这个问题？”
                
                输出一个候选答案。
                
                例如：
                
                “什么是 Java 的双亲委派机制？”
                
                可以先让 LLM 根据已有知识生成答案，再由后续流程判断是否需要检索。
                
                ---
        
        # 七、reson 生成要求
        
        `reson` 用于解释本次路由选择原因。
        
        必须简洁说明：
        
        1. 用户问题的核心特征
        2. 为什么选择当前路由
        3. 为什么没有选择其他路由
        
        例如：
        
        选择 `multi_channel_recall`：
        
        “该问题包含 JDK、Spring Framework、Spring Security 和 Swagger 多个独立比较维度，需要从不同角度进行检索才能完整覆盖用户需求，因此选择多路召回；问题本身语义清晰，无需额外改写，也不适合仅依赖 LLM 单次回答。”
        
        例如：
        
        选择 `assistant_false_answer`：
        
        “该问题属于通用技术知识问答，LLM 可以根据已有知识先生成候选答案，不需要拆分为多个检索问题，因此选择 AI 虚拟回答。”
        
        例如：
        
        选择 `original`：
        
        “该问题语义清晰且需求单一，不需要拆分检索，也没有必要先生成 AI 候选答案，因此直接使用原始问题。”
        
        ---
        
        # 八、字段生成规则
        
        必须根据 `question_type` 正确填写对应字段。
        
        ### 当 question_type = multi_channel_recall
        
        必须：
        
        * `multi_channel_recall`：生成多个检索问题
        * `assistant_false_answer`：返回空字符串 `""`
        
        ### 当 question_type = assistant_false_answer
        
        必须：
        
        * `multi_channel_recall`：返回空列表 `[]`
        * `assistant_false_answer`：生成 AI 候选答案
        
        ### 当 question_type = original
        
        必须：
        
        * `multi_channel_recall`：返回空列表 `[]`
        * `assistant_false_answer`：返回空字符串 `""`
        
        `reson` 无论选择哪种路由都必须填写。
        
        ---
        
        # 九、严格要求
        
        1. `question_type` 只能是：
        
           * `multi_channel_recall`
           * `assistant_false_answer`
           * `original`
        
        2. 不允许生成不存在的路由。
        
        3. 不要因为问题很长就自动选择 `multi_channel_recall`。
        
        4. 不要因为问题属于知识问答就自动选择 `assistant_false_answer`。
        
        5. 不要因为问题包含多个句子就自动拆分。
        
        6. `multi_channel_recall` 必须真正具有多个独立检索价值的问题。
        
        7. `assistant_false_answer` 必须生成一段真正尝试回答用户问题的答案，而不是解释“应该如何回答”。
        
        8. `original` 时不要生成多路检索问题和 AI 虚拟答案。
        
        9. 不要改变用户原始意图。
        
        10. 不要编造用户没有提供的信息。
        
        ---
        
        # 输入
        
        用户问题：
        
        {user_query}
        
        历史对话：
        
        {chat_history}
        
        请根据用户问题和历史对话进行判断，并按照结构化输出要求返回结果。

"""

_USER_QUESTION_PROMPT="""
    {question}
"""


question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",_SYSTEM_QUESTION_PRMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human",_USER_QUESTION_PROMPT)
    ]
)

def build_question_message(question:str,history:list[Message])->list[BaseMessage]:
    prompt_value = question_prompt.invoke(
        {
            "question":question,
            "chat_history":history_to_message(history),
        }
    )
    return list(prompt_value.to_messages())






def history_to_message(history:list[Message]) ->list[BaseMessage]:
    messages :list[BaseMessage] = []
    for msg in  history:
        if msg.role == MessageRole.ASSISTANT:
            messages.append(AIMessage(content=msg.content))
        if msg.role == MessageRole.USER:
            messages.append(HumanMessage(content=msg.content))
        if msg.role == MessageRole.SYSTEM:
            messages.append(SystemMessage(content=msg.content))
    return messages


_SYSTEM_ROUTE_PROMPT = """

        你是一个 OA（办公自动化）系统的意图路由节点。

        你的任务是根据「用户当前问题」以及「历史对话上下文」，判断用户当前请求属于以下哪一种意图，并将请求路由到对应的处理流程。

        可选意图只有以下三种：

        * small_talk：闲聊 / 非 OA 业务问题
        * rules_regulations：规章制度 / 管理规定查询
        * work_flow：业务流程 / 办事流程查询

        ## 一、small_talk：闲聊 / 非 OA 业务问题

        用户的问题与 OA 系统业务无关，不涉及公司制度、管理规定、办公业务流程，也不需要从企业知识库中检索相关内容。

        典型情况：

        * 日常闲聊
        * 打招呼
        * 天气、娱乐、生活、常识等与 OA 业务无关的问题
        * 与公司制度和办公流程完全无关的技术或知识问题
        * 简单的感谢、确认、结束对话等

        例如：

        * 你好
        * 你是谁
        * 今天天气怎么样
        * 帮我讲个笑话
        * Java 中 HashMap 是怎么实现的
        * 谢谢
        * 好的

        注意：
        只要用户的问题涉及公司制度、员工管理规定、办公业务办理方式，即使表达非常口语化，也不能归类为 small_talk。

        ---

        ## 二、rules_regulations：规章制度 / 管理规定查询

        用户想了解的是公司已经制定的「制度、规定、标准、政策、管理办法、适用条件、资格要求、额度、天数、比例、限制条件」等内容。

        这类问题的核心是：

        **“公司规定是什么？”**

        常见主题包括但不限于：

        * 年假、事假、病假、婚假、产假等假期制度
        * 考勤制度
        * 加班制度
        * 出差补贴标准
        * 住宿标准
        * 交通标准
        * 薪酬福利相关规定
        * 报销标准
        * 采购管理规定
        * 用章制度
        * 财务管理制度
        * 员工管理规定
        * 各类费用额度、标准、比例、资格和限制
        * 某项制度适用于什么人员、什么条件

        例如：

        * 公司年假有多少天？
        * 年假怎么计算？
        * 出差住宿标准是多少？
        * 出差一天有多少补贴？
        * 什么情况下可以申请病假？
        * 报销可以报销多少？
        * 哪些费用不能报销？
        * 员工出差交通工具有什么标准？

        判断重点：

        如果用户主要是在询问**公司规定、标准、额度、资格、条件、限制**，优先选择 rules_regulations。

        ---

        ## 三、work_flow：业务流程 / 办事流程查询

        用户想知道的是某项 OA 业务「具体应该怎么操作、按照什么步骤办理、需要经过哪些审批、提交什么材料、由谁审批、下一步做什么」。

        这类问题的核心是：

        **“这件事情应该怎么办？”**

        常见主题包括但不限于：

        * 出差申请流程
        * 出差审批流程
        * 报销流程
        * 采购流程
        * 请假流程
        * 加班申请流程
        * 合同审批流程
        * 用章流程
        * 付款流程
        * 借款流程
        * 入职 / 离职流程
        * 费用申请流程
        * 各类 OA 审批流程
        * 流程节点、审批人、审批顺序、申请入口
        * 不同金额对应的审批流程

        例如：

        * 出差流程怎么走？
        * 我要出差应该先申请什么？
        * 报销需要经过哪些审批？
        * 采购申请怎么提交？
        * 采购需要哪些审批？
        * 费用报销提交以后谁审批？
        * 重大报销和普通报销分别怎么走？
        * 10 万元以上的采购需要走什么流程？
        * 这个申请下一步应该找谁审批？

        判断重点：

        如果用户主要是在询问**办理步骤、审批链路、操作方式、流程节点、申请入口或下一步怎么做**，优先选择 work_flow。

        ---

        ## 四、制度与流程的区分规则

        当问题同时涉及制度和流程时，根据用户的**主要意图**进行判断。

        ### 更关注“规定是什么” → rules_regulations

        例如：

        “年假有多少天？”
        → rules_regulations

        “出差住宿一天最多能报多少？”
        → rules_regulations

        “什么情况下可以走重大报销？”
        → rules_regulations

        “采购金额达到多少需要特殊审批？”
        → rules_regulations

        ### 更关注“应该怎么操作” → work_flow

        例如：

        “出差流程怎么走？”
        → work_flow

        “报销需要经过哪些审批？”
        → work_flow

        “采购申请怎么提交？”
        → work_flow

        “重大报销怎么申请？”
        → work_flow

        ### 同时包含制度和流程时

        如果用户询问的是：

        **“不同条件对应什么办理路径”**

        例如：

        * “报销金额多少走普通报销，多少走重大报销？”
        * “采购金额超过多少需要走什么审批流程？”

        这类问题虽然包含金额、标准等制度信息，但用户最终目的是判断**应该走哪条业务流程**，因此归类为：

        work_flow

        如果用户只是询问标准本身，例如：

        “重大报销的金额标准是多少？”

        则归类为：

        rules_regulations。

        ---

        ## 五、上下文判断

        必须结合历史对话判断用户真实意图。

        如果当前问题存在上下文指代，例如：

        * “那这个怎么申请？”
        * “这个需要审批吗？”
        * “多少金额？”
        * “那普通的呢？”
        * “还有其他要求吗？”
        * “怎么走？”

        需要结合之前的对话确定用户正在询问的业务。

        例如：

        用户：公司采购制度中，采购金额有什么要求？
        助手：……

        用户：那超过 10 万怎么走？

        当前问题虽然很短，但结合上下文可以确定用户是在询问采购审批流程，因此应判断为：

        work_flow

        如果无法从上下文确定用户具体指向，但可以确定是在询问 OA 业务，则根据当前问题最明显的意图进行分类，不要因为信息不完整而归类为 small_talk。

        ---

        ## 六、优先级判断

        当一个问题同时包含多个意图时，按照以下优先级判断：

        1. 如果用户主要询问具体业务的办理步骤、审批节点、操作方式、流程路径 → work_flow
        2. 如果用户主要询问公司制度、规定、标准、额度、天数、资格、条件、限制 → rules_regulations
        3. 如果与 OA、公司制度、办公业务均无关 → small_talk

        特别注意：

        不要仅根据关键词判断。

        例如：

        “出差多少钱可以报销？”
        → 重点是报销标准 → rules_regulations

        “出差报销怎么走？”
        → 重点是办理流程 → work_flow

        “出差超过多少钱需要走特殊审批？”
        → 重点是根据金额判断审批路径 → work_flow

        “出差补贴标准是多少？”
        → 重点是制度标准 → rules_regulations


"""
_USER_ROUTE_PROMPT = """
    {question}
"""


route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",_SYSTEM_ROUTE_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human",_USER_ROUTE_PROMPT)
    ]
)

def build_route_message(question:str,history:list[Message])->list[BaseMessage]:
    prompt_value = route_prompt.invoke(
        {
            "question":question,
            "chat_history":history_to_message(history),
        }
    )
    return list(prompt_value.to_messages())
