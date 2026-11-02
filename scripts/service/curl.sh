# script to test model key for DO serverless access 
# pre-req: set env variable DO_MODEL_KEY on terminal 
# export DO_MODEL_KEY 
# Then run this script 


curl https://inference.do-ai.run/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $DO_MODEL_KEY" \
-d '{
  "model": "anthropic-claude-sonnet-5.5",
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of Bihar?"
    }
  ],
  "max_tokens": 100
}'

