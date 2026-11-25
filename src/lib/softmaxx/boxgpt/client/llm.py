from openai import OpenAI
from typing import Final
import logging 
from softmaxx.trace import Trace
from tqdm import tqdm


class DigitalOceanClient:
    SONNET_55_MODEL : Final[str] = "anthropic-claude-sonnet-5.5"
    BAAI_BGE_M3 : Final[str] = "bge-m3"
    DEEPSEEK_v41_FLASH_1017: Final[str] = "deepseek-v4.1-flash-1017"

    def __init__(self, api_key:str):
        """Initializes the OpenAI client."""
        if not api_key:
            raise ValueError("api key is required")
            
        self.client = OpenAI(base_url="https://inference.do-ai.run/v1/", 
            api_key=api_key, 
            timeout=60.0
        )

    def chat_stream(self, 
        model: str, 
        llm_messages: list[dict[str, str]],
        temperature: float=0,
        verbose: bool=False
    )-> str:
        stream = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=llm_messages,
            stream=True
        )

        logger = logging.getLogger("main." + __name__)
        trace = Trace(logger, enabled=verbose)
        chunks: list[str] = []
        content_chunk_count = 0

        try:
            with tqdm(desc="LLM stream", bar_format="{desc}: chunk [{n}] ", leave=True) as pbar:
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        content_chunk_count += 1
                        chunks.append(delta)
                        pbar.update(1)

            trace.log(f"LLM stream : received {content_chunk_count} chunks")

        finally:
            stream.close()

        return "".join(chunks)
        
       
    def sonnet_55_chat(self, llm_messages:list[dict[str, str]]):
        response = self.client.chat.completions.create(
            model="anthropic-claude-sonnet-5.5",
            messages=llm_messages,
            temperature=0.0,
            max_tokens=1000
        )

        answer = response.choices[0].message.content
        return answer

    
    def bgem3_embeddings(self, doc_chunks:list[str]):
        response = self.client.embeddings.create(
            model="bge-m3",
            input=doc_chunks,
            encoding_format="float",
        )
        
        return response
