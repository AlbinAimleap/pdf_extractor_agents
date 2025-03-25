from pydantic_ai import RunContext
from  typing import List

def vector_search(ctx: RunContext, query: str, top_k: int) -> List[str]:
    result = ctx.deps.search(query, top_k=top_k)
    return result
    







