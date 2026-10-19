"""
答案生成节点（node_answer_output）：组装 Prompt → 生成答案 → 引用溯源 → 历史落库。
"""
from app.process.query.agent.state import QueryGraphState
from app.rag.query import produce_answer
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_answer_output")
def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    """生成最终答案（按角色模板 + 合规提醒 + 来源引用）。"""
    add_running_task(state["session_id"], "node_answer_output", state.get("is_stream"))
    state = produce_answer(state)
    add_done_task(state["session_id"], "node_answer_output", state.get("is_stream"))
    return state
