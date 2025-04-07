
# RAG - Elasticsearch with Ollama/LM Studio

## Application Requirements

This application relies on two services running simultaneously to function properly:

### LLM Server  
You can use one or both of the following options to serve the language model:

- **Ollama** - [https://ollama.com](https://ollama.com)  
  The model server must be running at: `http://localhost:11434/v1`

- **LM Studio** - [https://lmstudio.ai](https://lmstudio.ai)  
  The model server must be running at: `http://localhost:1234/v1`

### Elasticsearch Server  
You need to have an Elasticsearch instance running locally.  
Installation instructions can be found here:  
[https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html)

To quickly set up Elasticsearch and Kibana locally, run the following script:

```bash
curl -fsSL https://elastic.co/start-local | sh
```


### Instalação

```bash
  pip install -r requirements.txt
```

```bash
  uvicorn api:app --reload --port 8080 
```
    