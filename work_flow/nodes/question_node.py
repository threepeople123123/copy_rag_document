from pydantic import Field, BaseModel

from llm.models import get_agent, get_chat_model
from llm.prompts import build_question_message
from work_flow.copy_rag_document_state import CopyRagDocumentState, question_choose


class QuestionDiverse(BaseModel):

    # 路由选择
    question_type : question_choose = Field(..., description="路由选择，multi_channel_recall,assistant_false_answer,original")

    # ai生成的路由召回问题
    multi_channel_recall:list[str] = Field(..., description="生成多角度提问的回答")

    # 选择说明
    reason:str= Field(..., description="说明，为什么这样做")

    # ai虚拟回答进行召回
    assistant_false_answer:str = Field(..., description="llm生成的答案")

async def route_nodes(state:CopyRagDocumentState)->CopyRagDocumentState:


    question_rewrite = state["question_rewrite"]
    history = state.get("history",[])

    message = build_question_message(question_rewrite,history)

    model_with_structure  = get_chat_model().with_structured_output(QuestionDiverse)

    question_diverse = await model_with_structure.ainvoke(message)

    update : CopyRagDocumentState = {"question_route":"original"}

    if question_diverse.question_type ==  "multi_channel_recall":

        multi_channel_recall = question_diverse.multi_channel_recall
        update["multi_channel_recall"] = multi_channel_recall
        update["question_route"] = "multi_channel_recall"

    elif question_diverse.question_type ==  "assistant_false_answer":

        assistant_false_answer = question_diverse.assistant_false_answer
        update["assistant_false_answer"] = assistant_false_answer
        update["question_route"] = "assistant_false_answer"

    return update