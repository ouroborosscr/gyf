def mark_incomplete(state: dict) -> dict:
    return {**state, "packet_incomplete": True}