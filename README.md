# Chat con PDF Local (RAG) usando Ollama y LangChain

Este proyecto permite preguntar sobre archivos PDF de forma local y privada, utilizando modelos de lenguaje (LLM) que corren en tu propia máquina, a **Ollama**. No se envían datos a la nube.

## Requisitos previos

Antes de empezar, asegúrate de tener instalado **Python 3.11** o superior en tu sistema (probado en Linux).

## Instalación y ejecución 

Ollama es el motor que permite ejecutar los modelos de IA localmente.

1. **Descarga e instalación en Linux:**
   Abre una terminal y ejecuta el script oficial:
   ```bash
   curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
   ```
   
  Descarga de los modelos necesarios: 
  ```bash
  ollama pull phi3
  ollama pull nomic-embed-text
  ```

  Verificar que Ollama está activo:
  ```bash
  ollama list
  ```

2. **Dependencias de Python:**
 ```bash
 pip install langchain langchain-community langchain-ollama langchain-chroma pypdf
```

3. **Ejecutar script:**
  ```bash
  python main.py
  ```
  
