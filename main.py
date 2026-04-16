from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Cargar y partir el PDF
# Asegúrate de que el nombre del archivo sea correcto
path = input("\n Indica un directorio con la ruta completa (debe terminar con '/') $: ")
file = input("\n Indica el nombre del pdf $: ")
loader = PyPDFLoader(path + file)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 2. Base de datos vectorial (Usando Ollama)
print("--- Creando base de datos vectorial local ---")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# 3. El Cerebro (LLM)
llm = OllamaLLM(model="phi3")

# 4. Definir la lógica del Chat (Sin usar el problemático RetrievalQA)
template = """Responde basándote solo en el siguiente contexto:
{context}

Pregunta: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Esta es la "cadena" moderna. No necesita el import 'langchain.chains'
rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# 5. Ejecución
print("--- PDF Cargado. IA lista. ---")

while True:
    # Captura la pregunta del usuario
    pregunta = input("\n👤 Tú: ")

    # Condición para salir del bucle
    if pregunta.lower() in ["salir", "exit", "quit"]:
        print("Cerrando el chat. ¡Hasta luego!")
        break

    if not pregunta.strip():
        continue

    print("🤖 Pensando...")

    try:
        # Invocamos la cadena con la pregunta del prompt
        respuesta = rag_chain.invoke(pregunta)
        print(f"\n🤖 IA: {respuesta}")
    except Exception as e:
        print(f"❌ Ocurrió un error: {e}")
