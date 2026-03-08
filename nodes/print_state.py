from state import State
from utils.config import ENABLE_PRINT
import logging

def print_state(state: State):
    logging.info("进入 print_state 节点")
    if ENABLE_PRINT:
        print(state["is_suspicious"])
        print(state["suspicious_flows_start"])
        print(state["suspicious_flows_end"])
        print(state["suspicious_flows"])




    return 