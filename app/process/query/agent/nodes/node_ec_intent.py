"""
意图识别节点（node_ec_intent）：问题改写 + 领域筛选条件抽取 + 合规风险检测。
"""
from app.process.query.agent.state import QueryGraphState
from app.rag.query import confirm_ec_intent
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_ec_intent")
def node_ec_intent(state: QueryGraphState) -> QueryGraphState:
    """识别跨境电商问题意图，改写查询并抽取筛选条件。"""
    add_running_task(state["session_id"], "node_ec_intent", state.get("is_stream"))
    state = confirm_ec_intent(state)
    add_done_task(state["session_id"], "node_ec_intent", state.get("is_stream"))
    return state


if __name__ == "__main__":
    from app.process.query.agent.state import create_query_default_state

    mock_state = create_query_default_state(
        session_id="test_intent_001",
        original_query="亚马逊FBA仓储费太高，滞销库存怎么清理？",
        is_stream=False,
    )
    try:
        result = node_ec_intent(mock_state)
        print(f"原始问题: {result['original_query']}")
        print(f"改写后: {result['rewritten_query']}")
        print(f"筛选条件: {result['ec_filters']}")
        print(f"风险等级: {result['risk_level']} {result['risk_types']}")
    except Exception as exc:
        print(f"测试失败（需配置 LLM）: {exc}")
