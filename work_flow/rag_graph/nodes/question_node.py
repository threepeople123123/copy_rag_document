from pydantic import Field, BaseModel

from core.langfuse import send_message_to_langfuse
from llm.models import get_structured_agent
from llm.prompts import build_question_message
from work_flow.rag_graph.copy_rag_document_state import RagDocumentState, question_choose


class QuestionDiverse(BaseModel):

    # 路由选择
    question_type : question_choose = Field(..., description="路由选择，multi_channel_recall,assistant_false_answer,original")

    # ai生成的路由召回问题
    multi_channel_recall:list[str] = Field(..., description="生成多角度提问的回答")

    # 选择说明
    reason:str= Field(..., description="说明，为什么这样做")

    # ai虚拟回答进行召回
    assistant_false_answer:str = Field(..., description="llm生成的答案")

async def question_node(state:RagDocumentState)->RagDocumentState:


    question_rewrite = state["question_rewrite"]
    history = state.get("history",[])

    message = build_question_message(question_rewrite,history)

    trace_id = state["trace_id"]
    # 重写后问题

    input_message:dict = {}
    output_message:dict = {}

    # 结构化agent
    agent_with_structure = get_structured_agent(QuestionDiverse)
    result = await agent_with_structure.ainvoke({
        "messages": message,
    })
    question_diverse = result["structured_response"]

    update : RagDocumentState = {"question_route": "original"}

    input_message={"question": question_rewrite}

    if question_diverse.question_type ==  "multi_channel_recall":

        multi_channel_recall = question_diverse.multi_channel_recall
        update["multi_channel_recall"] = multi_channel_recall
        update["question_route"] = "multi_channel_recall"

        output_message={"question_route":"multi_channel_recall","ai_generate": multi_channel_recall}

    elif question_diverse.question_type ==  "assistant_false_answer":

        assistant_false_answer = question_diverse.assistant_false_answer
        update["assistant_false_answer"] = assistant_false_answer
        update["question_route"] = "assistant_false_answer"

        output_message={"question_route":"assistant_false_answer","ai_generate": assistant_false_answer}
    # 发送langfuse消息
    await send_message_to_langfuse(trace_id,__name__,input_message,output_message)

    return update