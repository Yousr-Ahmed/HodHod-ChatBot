import os
from decouple import config

# Set this to `azure`
os.environ["OPENAI_API_TYPE"] = config("OPENAI_API_TYPE", default="azure")
# The API version you want to use: set this to `2023-07-01` for the released version.
os.environ["OPENAI_API_VERSION"] = config("OPENAI_API_VERSION", default="2023-07-01-preview")
# The base URL for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
os.environ["OPENAI_API_BASE"] = config("OPENAI_API_BASE", default="")
# The API key for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY", default="")

os.environ["SERPAPI_API_KEY"] = config("SERPAPI_API_KEY", default="")

os.environ["DEPLOYMENT_NAME"] = config("DEPLOYMENT_NAME", default="")

persist_directory = "db"
azure_embeddings_deployment_name = config("AZURE_EMBEDDINGS_DEPLOYMENT_NAME", default="")
