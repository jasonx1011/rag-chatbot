# rag-chatbot
LLM chatbot based on Retrieval Augmented Generation (RAG)

### Virtual env & install packages (Python3.10+)###
$virtualenv venv  
$source ./venv/bin/activate  
$(venv) pip install requirements.txt  

### Build vec index (optional)###
using download embedding model:  
$python build_vec.py -m openai_emb  
using OpenAI embedding model:  
$python build_vec.py -m openai_emb -k <OpenAI API key>  

### Run streamlit ###
$streamlit run chat.py  
