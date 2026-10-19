"""
查询流程主图（main_graph）：组织问答链节点执行顺序与分支流转。

链路：
  node_ec_intent ──(已直接产出答案: 寒暄/澄清)──> node_answer_output
                └─(需检索)──> node_search → node_rrf → node_rerank → node_answer_output → END
"""
from langgraph.graph import END, StateGraph

from app.process.query.agent.nodes.node_answer_output import node_answer_output
from app.process.query.agent.nodes.node_ec_intent import node_ec_intent
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_search import node_search
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger

# 1. 定义状态图
query_graph = StateGraph(QueryGraphState)

# 2. 添加节点
query_graph.add_node("node_ec_intent", node_ec_intent)
query_graph.add_node("node_search", node_search)
query_graph.add_node("node_rrf", node_rrf)
query_graph.add_node("node_rerank", node_rerank)
query_graph.add_node("node_answer_output", node_answer_output)

# 3. 入口节点
query_graph.set_entry_point("node_ec_intent")


# 4. 意图识别后条件路由
def after_intent_router(state: QueryGraphState):
    """寒暄/澄清分支已直接产出 answer，跳过检索直达答案生成。"""
    if state.get("answer"):
        logger.info("意图节点已直接产出答案（寒暄/澄清），跳过检索")
        return "node_answer_output"
    return "node_search"


query_graph.add_conditional_edges(
    "node_ec_intent",
    after_intent_router,
    {
        "node_answer_output": "node_answer_output",
        "node_search": "node_search",
    },
)

# 5. 静态边
query_graph.add_edge("node_search", "node_rrf")
query_graph.add_edge("node_rrf", "node_rerank")
query_graph.add_edge("node_rerank", "node_answer_output")
query_graph.add_edge("node_answer_output", END)

# 6. 编译
query_app = query_graph.compile()

if __name__ == "__main__":
    from app.process.query.agent.state import create_query_default_state

    logger.info("===== 问答链测试 =====")
    state = create_query_default_state(
        session_id="test_query_001",
        original_query="亚马逊FBA仓储费怎么计算？",
        is_stream=False,
    )
    try:
        result = query_app.invoke(state)
        print(f"问题: {result['original_query']}")
        print(f"改写: {result['rewritten_query']}")
        print(f"筛选: {result['ec_filters']}")
        print(f"答案: {(result.get('answer') or '')[:200]}")
    except Exception as exc:
        print(f"测试失败（需配置 LLM/向量库）: {exc}")
