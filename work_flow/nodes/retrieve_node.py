from work_flow.copy_rag_document_state import CopyRagDocumentState


async def retrieve_node(state:CopyRagDocumentState)->CopyRagDocumentState:

    question_route = state["question_route"]
    multi_channel_recall = state["multi_channel_recall"]
    assistant_false_answer = state["assistant_false_answer"]

    if question_route == "original":
        ...
    if question_route == "multi_channel_recall":
        #多路召回
        ...
    elif question_route == "assistant_false_answer":
        # 重写问题查询一次,ai返回查询一次,.两次查询,跟多路召回统一逻辑
        ...



    return {}


def rrf_sort():
    ...
