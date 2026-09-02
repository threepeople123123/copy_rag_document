from uuid import UUID

from search.models import RetrieveChunk
from search.search import Search
from work_flow.copy_rag_document_state import CopyRagDocumentState


async def retrieve_node(state:CopyRagDocumentState)->CopyRagDocumentState:

    question_route = state["question_route"]
    multi_channel_recall = state.get("multi_channel_recall",[])
    assistant_false_answer = state.get("assistant_false_answer","")
    question_rewrite = state.get("question_rewrite",state["question"])

    search = Search()

    multi_channel_recall_result: list[list[RetrieveChunk]] = []

    chunks:list[RetrieveChunk] = []

    if question_route == "original":

        chunks = await search.search_all(question_rewrite,20)
    if question_route == "multi_channel_recall":
        #多路召回
        for multi_question in multi_channel_recall:
            chunks = await search.search_all(multi_question,20)
            multi_channel_recall_result.append(chunks)
        chunks = merge(multi_channel_recall_result,20)
    elif question_route == "assistant_false_answer":
        # 重写问题查询一次,ai返回查询一次,.两次查询,跟多路召回统一逻辑
        multi_channel_recall_result.append(await search.search_all(assistant_false_answer, 20))
        multi_channel_recall_result.append(await search.search_all(question_rewrite, 20))
        chunks = merge(multi_channel_recall_result,20)


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
