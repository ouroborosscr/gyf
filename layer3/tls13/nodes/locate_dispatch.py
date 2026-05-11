def locate_dispatch(state: dict) -> dict:
    """根据 case_tried 决定下一步走哪个 case。
    路由由 graph.py 中的 add_conditional_edges 实现。
    """
    tried = state.get("case_tried", [])
    if "case1" not in tried:
        return {**state, "_next": "case1"}
    if "case2" not in tried:
        return {**state, "_next": "case2"}
    if "case3" not in tried:
        return {**state, "_next": "case3"}
    # 三种都试过了
    if state.get("extract_ok"):
        return {**state, "_next": "threat_intel"}
    return {**state, "_next": "mark_incomplete"}


def route_after_dispatch(state: dict) -> str:
    return state.get("_next", "mark_incomplete")