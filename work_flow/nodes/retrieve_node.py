from uuid import UUID

from core.langfuse import langfuse_client, send_message_to_langfuse
from search.models import RetrieveChunk
from search.search import Search
from work_flow.copy_rag_document_state import CopyRagDocumentState


async def retrieve_node(state:CopyRagDocumentState)->CopyRagDocumentState:

    question_route = state["question_route"]
    multi_channel_recall = state.get("multi_channel_recall",[])
    assistant_false_answer = state.get("assistant_false_answer","")
    question_rewrite = state.get("question_rewrite",state["question"])
    trace_id = state["trace_id"]

    search = Search()

    multi_channel_recall_result: list[list[RetrieveChunk]] = []

    chunks:list[RetrieveChunk] = []

    input_message:dict = {}
    output_message:dict = {}

    if question_route == "original":
        chunks = await search.search_all(question_rewrite,20)
        output_message={"retrieve":str(chunks)}
        input_message={"question": question_rewrite,"route": "original"}

    if question_route == "multi_channel_recall":
        #多路召回
        for multi_question in multi_channel_recall:
            chunks = await search.search_all(multi_question,20)
            multi_channel_recall_result.append(chunks)
        multi_channel_recall_result.append(await search.search_all(question_rewrite, 20))
        chunks = merge(multi_channel_recall_result,20)

        output_message={"retrieve": str(chunks)}
        input_message={"question": str(multi_channel_recall),"route": "multi_channel_recall"}
    elif question_route == "assistant_false_answer":
        # 重写问题查询一次,ai返回查询一次,.两次查询,跟多路召回统一逻辑
        multi_channel_recall_result.append(await search.search_all(assistant_false_answer, 20))
        multi_channel_recall_result.append(await search.search_all(question_rewrite, 20))
        chunks = merge(multi_channel_recall_result,20)
        output_message={"retrieve": str(chunks)}
        input_message={"question": f"ai回答：{assistant_false_answer},重写问题：{question_rewrite}","route": "assistant_false_answer"}

    # 发送langfuse
    await send_message_to_langfuse(trace_id,"retrieve_node",input_message,output_message,as_type="retriever")

    return {"retrieve":chunks}


def merge(merge_list_list: list[list[RetrieveChunk]],top_k:int)->list[RetrieveChunk]:
    merge_result_map :dict[UUID, RetrieveChunk] = {}
    for merge_list in merge_list_list:
        for chunk in merge_list:
            result = merge_result_map.get(chunk.chunk_id)
            if result is not None:
                if chunk.rrf_score > result.rrf_score:
                    merge_result_map[chunk.chunk_id] = chunk
            else:
                merge_result_map[chunk.chunk_id] = chunk
    return sorted(merge_result_map.values(),key= lambda x : x.rrf_score,reverse=True)[:top_k]
