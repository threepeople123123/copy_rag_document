import json

from pydantic import Field, BaseModel, TypeAdapter

from core.langfuse import send_message_to_langfuse
from llm.models import get_structured_agent
from llm.prompts import build_plan_prompt
from work_flow.copy_rag_document_state import CopyRagDocumentState, PlanReason, intent_choose, question_choose


class PlanState(BaseModel):
    # 问题重写
    question_rewrite:str|None = Field(...,description="用户问题重写")

    # 问题重写
    multi_channel_recall: list[str] |None= Field(...,description="根据用户提问从多个角度提出问题，进行多路召回")

    # ai虚假回答
    assistant_false_answer:str|None = Field(...,description="ai根据问题")

    # 路由，查看是否需要重写，多路召回，ai假写，或者原始问题不用动
    question_route:question_choose|None = Field(...,description="路由，查看是否需要重写，多路召回，ai假写，或者原始问题不用动")

    # 意图识别，查看是否是闲聊，或者流程查询，还是规章制度查询
    intent:intent_choose|None = Field(...,description="意图识别，查看是否是闲聊，或者流程查询，还是规章制度查询")


async def plan_node(state:CopyRagDocumentState)->CopyRagDocumentState:
    #
    cycle_count = state.get("cycle_count", 1)

    plan_reasons = state.get("plan_reasons", [])

    history = state.get("history", [])

    trace_id = state["trace_id"]

    reason = build_plan_reason(state)


    plan_reasons.append(PlanReason(count=cycle_count,reason=reason))

    plan_reason_json = json.dumps(plan_reasons)

    message = build_plan_prompt(plan_reason_json,history)

    agent_with_structure = get_structured_agent(PlanState)
    result = await agent_with_structure.ainvoke({
        "messages":message,
    })

    plan_state_result = result["structured_response"]

    update:CopyRagDocumentState = {"cycle_count":cycle_count,"plan_reasons":plan_reasons}

    cycle_count = cycle_count + 1

    update["cycle_count"] = cycle_count
    update["question_rewrite"] = plan_state_result.question_rewrite
    update["multi_channel_recall"] = plan_state_result.multi_channel_recall
    update["assistant_false_answer"] = plan_state_result.assistant_false_answer
    update["question_route"] = plan_state_result.question_route
    update["intent"] = plan_state_result.intent


    json_str = json.dumps(plan_reasons, ensure_ascii=False, indent=4)
    input_message:dict = {"plan_reasons":json_str}
    output_message:dict = {
        "question_rewrite":plan_state_result.question_rewrite,
        "multi_channel_recall":plan_state_result.multi_channel_recall,
        "assistant_false_answer":plan_state_result.assistant_false_answer,
        "question_route":plan_state_result.question_route,
        "intent":plan_state_result.intent,
    }

    # 发送langfuse消息
    await send_message_to_langfuse(trace_id,"plan_node",input_message,output_message,"chain")

    return update


def build_plan_reason(state:CopyRagDocumentState)->str:

    cycle_count = state.get("cycle_count", 1)
    question = state["question"]
    relevance_score = state["relevance_score"]
    relevance_reason = state["relevance_reason"]
    intent = state["intent"]
    retrieve = state["retrieve"]
    question_rewrite = state.get("question_rewrite",state["question"])
    question_route = state["question_route"]

    chunks_list = [
        f"片段{index} : \n 内容：{chunk.content}"
        for index,chunk in enumerate(retrieve, start=1)
    ]

    chunk_json = json.dumps(chunks_list)
    return f"""
            第{cycle_count}轮循环，
            原始问题:{question}，\t 重写问题:{question_rewrite}，
            问题路由：{question_route}(路由分为: multi_channel_recall:ai生成多个问题，进行多路召回，assistant_false_answer：爱生成虚假回答，进行召回，original，使用重写后的问题，没有则使用原始提问)
            意图识别:{intent},解释：因为这是oa系统，所有我将用户的意图识别为，闲聊：small_talk（例如：今天天气真好哦。这种不需要查询分片文档，ai直接回答），规章制度查询：rules_regulations（例如：入职一年年假有几天），流程审批查询：work_flow（例如：100万的采购审批应该走重大采购，还是普通采购），
            检索出来的文档:{chunk_json}
            \n
            文档相关性评估：分数：{relevance_score},理由：{relevance_reason},
                            0.9 ~ 1.0：文档包含回答用户问题所需的完整信息，可以直接、准确地回答。
                            0.8 ~ 0.89：文档与问题高度相关，包含主要答案，仅缺少少量非关键细节。
                            0.6 ~ 0.79：文档基本相关，可以回答问题的核心部分，但存在部分信息缺失。
                            0.4 ~ 0.59：文档与问题存在一定相关性，但缺少回答问题所需的关键内容，不能完整回答。
                            0.2 ~ 0.39：只有部分关键词或背景概念相关，无法有效回答用户问题。
                            0.0 ~ 0.19：文档与用户问题基本无关。
        """