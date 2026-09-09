from sys import prefix

from fastapi import APIRouter

from work_flow.test_graph.test_case_graph import get_test_graph

route = APIRouter(prefix="/test_case",tags=["test_case"])

@route.post("/generate",status_code=201)
async def generate_test_graph(question:str,conversation_id:str):
   config = {"configurable": {"thread_id": "thread-1"}}

   init = {"question":question}

   finally_state = await get_test_graph().ainvoke(input=init,config=config)
   return finally_state
