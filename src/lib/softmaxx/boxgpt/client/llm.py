from openai import OpenAI


class DigitalOceanClient:
    def __init__(self, api_key:str):
        """Initializes the OpenAI client."""
        if not api_key:
            raise ValueError("api key is required")
            
        self.client = OpenAI(base_url="https://inference.do-ai.run/v1/", 
                        api_key=api_key, 
                        timeout=60.0
                    )
    
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
