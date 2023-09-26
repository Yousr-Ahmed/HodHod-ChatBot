# @title Import libraries
import os
import shutil
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
from langchain.agents import AgentType, Tool, initialize_agent, load_tools
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pydantic import BaseModel

# @title Save Environment Variables
# Set this to `azure`
os.environ["OPENAI_API_TYPE"] = "azure"
# The API version you want to use: set this to `2023-07-01` for the released version.
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
# The base URL for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
os.environ["OPENAI_API_BASE"] = "https://hodhod-gpt.openai.azure.com/"
# The API key for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
os.environ["OPENAI_API_KEY"] = "d4eb4cb7f64646adb412ef89799ad89b"

os.environ[
    "SERPAPI_API_KEY"
] = "0f5445227ca55099fe1b2bff1d58e3cd1731ad999510d2f1eac5315cb393e7f3"

os.environ["DEPLOYMENT_NAME"] = "Hodhod_Assistant"

persist_directory = "db"
azure_embeddings_deployment_name = "ADA-Hodhod"

model = AzureChatOpenAI(
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
)
embeddings = OpenAIEmbeddings(deployment=azure_embeddings_deployment_name)
tools = load_tools(["serpapi", "llm-math"], llm=model)

# Print number of txt files in directory
loader = DirectoryLoader("", glob="Documents/*.*")
documents = loader.load()

# Splitting the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

print(
    f"Number of Documents = {len(documents)}",
    f"Number of Chunks = {len(texts)}",
    sep="\n",
)

vector_db = Chroma.from_documents(
    documents=texts, embedding=embeddings, persist_directory=persist_directory
)

vector_db_search = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=vector_db.as_retriever(),
    verbose=True,
    return_source_documents=True,
    input_key="question",
)

tools.append(
    Tool(
        name="Obeikan QA System",
        func=lambda query: vector_db_search({"question": query}),
        description="useful for when you need to answer questions about the Obeikan. Input should be a fully formed question. Output will be include the source document.",
    ),
)

# Define the API
app = FastAPI()


class Message(BaseModel):
    id: int
    msg: str


dic = {}


@app.post("/new_chat")
def new_chat():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_chain = initialize_agent(
        tools,
        model,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
    )
    new_id = len(dic) + 1
    dic[new_id] = agent_chain
    return {"id": f"{new_id}"}


@app.post("/new_msg")
def new_msg(input_msg: Message):
    agent_chain = dic[input_msg.id]
    response = agent_chain.run(input_msg.msg)
    return {"response": f"{response}"}


# Add a post method to the API that return a list of all files in the Documents folder using os.listdir()
@app.post("/view_files")
def view_files():
    return {"files": os.listdir("Documents")}


@app.post("/upload_files")
async def upload_files(files: List[UploadFile] = File(...)):
    files_name = [file.filename for file in files]
    for file in files:
        with open(f"Documents/{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    return {"detail": f"Files {', '.join(files_name)} uploaded successfully"}


# Add a post method to the API that delete file from the Documents folder using os.remove()
@app.post("/delete_files")
def delete_files(file_name: str):
    if file_name in os.listdir("Documents"):
        os.remove(f"Documents/{file_name}")
        return {"detail": f"File {file_name} deleted successfully"}
    return {"detail": f"File {file_name} not found"}
