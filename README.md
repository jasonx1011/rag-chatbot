# rag-chatbot
LLM chatbot based on Retrieval Augmented Generation (RAG)

### [online demo](https://rag-chatbot.streamlit.app/)
![Link Name](./assets/chat.png)

### Virtual env & install packages (with Python 3.10+)  
```bash
$ virtualenv venv  
$ source ./venv/bin/activate  
$ pip install requirements.txt  
```

### Knowledge base data & build vec index (optional)
You need to provide a csv file with header of ['link', 'title', 'body']  
'body' column is html format, but it will convert to markdown format later  
* using OpenAI embedding model (default):  
```bash
$ python build_vec.py -m openai_emb -k OpenAI API key  
```
* using downloaded embedding model:  
```bash
$ python build_vec.py -m BAAI/bge-small-en-v1.5 
```

### Run streamlit
```bash
$ streamlit run chat.py  
```
