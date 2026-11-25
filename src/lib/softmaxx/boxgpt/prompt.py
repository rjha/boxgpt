from pathlib import Path
from softmaxx.config import AppConfig


def rag_query_message(user_query, chunks):
    resource_dir = Path(AppConfig.get("resource_dir")).resolve()
    rag_prompt_dir = resource_dir / "prompts" / "rag"
    
    system_prompt = (rag_prompt_dir / "system.txt").read_text(encoding="utf-8").strip()
    user_template = (rag_prompt_dir / "user.txt").read_text(encoding="utf-8")
    
    # Build structured context string from MMR chunks
    context_builder = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("doc_name", "No Document")
        section = chunk.get("section_title", "General")
        content = chunk.get("content", "").strip()
        
        context_builder.append(
            f"[Snippet {i} | Source: {source} | Section: {section}]\n{content}\n"
        )
    
    formatted_context = "\n".join(context_builder)

    # Format the user prompt template with context and query
    formatted_user_prompt = user_template.format(
        context=formatted_context,
        user_query=user_query
    )

    return  [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": formatted_user_prompt}
    ]

