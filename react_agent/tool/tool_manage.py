import asyncio
import json
from typing import Any

from react_agent.react_state import ToolCall, ToolCallResponse
from react_agent.tool.registry import ToolRegistry, get_tool_registry


def _resolve_args(args: Any, results: dict[str, ToolCallResponse]) -> Any:
    """把参数里的 '{<tool_call_id>}' 占位符替换成前置工具结果（串行的数据传递）。"""
    if isinstance(args, str):
        for tid, r in results.items():
            args = args.replace("{" + tid + "}", r.response)
        return args
    if isinstance(args, list):
        return [_resolve_args(a, results) for a in args]
    if isinstance(args, dict):
        return {k: _resolve_args(v, results) for k, v in args.items()}
    return args


def _to_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, (dict, list)):
        return json.dumps(output, ensure_ascii=False)
    return str(output)


async def _run_one(call: ToolCall, registry: ToolRegistry,
                   results: dict[str, ToolCallResponse]) -> ToolCallResponse:
    tool = registry.get(call.name)
    if tool is None:
        return ToolCallResponse(
            id=call.id, name=call.name, ok=False,
            error=f"工具未注册: {call.name}",
        )

    args = _resolve_args(call.args, results)
    try:
        if hasattr(tool, "ainvoke"):
            output = await tool.ainvoke(args)
        else:
            output = await asyncio.to_thread(tool.invoke, args)
        return ToolCallResponse(
            id=call.id, name=call.name, response=_to_text(output), ok=True,
        )
    except Exception as e:
        return ToolCallResponse(
            id=call.id, name=call.name, ok=False, error=str(e),
        )


async def tool_manage(tool_calls: list[ToolCall],
                      registry: ToolRegistry | None = None) -> list[ToolCallResponse]:
    """统一调用入口：无依赖的工具并行，有依赖的串行（按依赖顺序）。"""
    registry = registry or get_tool_registry()

    results: dict[str, ToolCallResponse] = {}
    pending: dict[str, ToolCall] = {t.id: t for t in tool_calls}
    deps: dict[str, set[str]] = {
        t.id: set(t.depends_on or []) for t in tool_calls
    }

    while pending:
        ready = [t for t in pending.values() if deps[t.id] <= set(results)]
        if not ready:
            problem = [
                (t.id, sorted(deps[t.id] - set(results)))
                for t in pending.values() if deps[t.id] - set(results)
            ]
            raise RuntimeError(f"工具依赖无法满足（循环依赖或依赖不存在）: {problem}")

        batch = await asyncio.gather(*(_run_one(t, registry, results) for t in ready))
        for t, r in zip(ready, batch):
            results[t.id] = r
        for t in ready:
            del pending[t.id]

    return [results[t.id] for t in tool_calls]
