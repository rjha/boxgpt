from openai import OpenAI
from softmaxx.config import AppConfig



class DigitalOceanClient:
    def __init__(self, model_name: str):
        """Initializes the OpenAI client."""
        api_keys = AppConfig.get("api_keys")
        sonnet_api_key = api_keys[model_name]
        self.model_name = model_name
        self.client = OpenAI(base_url="https://inference.do-ai.run/v1/", 
                        api_key=sonnet_api_key, 
                        timeout=60.0
                    )
        
    def send_message(self, llm_messages):
        response = self.client.chat.completions.create(
            model="anthropic-claude-sonnet-5.5",
            messages=llm_messages,
            temperature=0.0,
            max_tokens=1000
        )

        answer = response.choices[0].message.content
        return answer

        
