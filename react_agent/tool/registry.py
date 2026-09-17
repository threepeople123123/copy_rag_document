from langchain_core.tools import BaseTool


class ToolRegistry:
    """统一工具注册中心：按 name 管理所有工具实例。"""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> "ToolRegistry":
        if not getattr(tool, "name", None):
            raise ValueError("工具必须定义 name")
        self._tools[tool.name] = tool
        return self

    def register_many(self, tools: list[BaseTool]) -> "ToolRegistry":
        for t in tools:
            self.register(t)
        return self

    def load_builtin(self, names: list[str]) -> "ToolRegistry":
        """批量加载内置/社区工具，如 ['llm-math', 'wikipedia']（按需懒加载，避免依赖缺失时拖垮整个模块）"""
        from langchain_community.agent_toolkits import load_tools
        return self.register_many(load_tools(names))

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def get_all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools


_registry: ToolRegistry | None = None


def register_all_tools() -> ToolRegistry:
    """集中注册项目所有工具（幂等，可重复调用）。"""
    from react_agent.tool.tools import (
        analyze_intention,
        parse_requirement,
        generate_test_cases,
        modify_test_cases,
        finish,
    )

    r = ToolRegistry()
    r.register(analyze_intention)
    r.register(parse_requirement)
    r.register(generate_test_cases)
    r.register(modify_test_cases)
    r.register(finish)
    return r


def get_tool_registry() -> ToolRegistry:
    """全局单例：首次调用时注册，之后复用。"""
    global _registry
    if _registry is None:
        _registry = register_all_tools()
    return _registry
