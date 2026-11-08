# script to test model key for DO serverless access 
# pre-req: set env variable DO_MODEL_KEY on terminal 
# export DO_MODEL_KEY 
# Then run this script 



curl "https://inference.do-ai.run/v1/embeddings"  \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DO_MODEL_KEY" \
  -d '{
    "model":"bge-m3",
    "input":["hello world","goodbye world"],
    "encoding_format":"float"
  }'
  

  