import os
os.environ["GOOGLE_API_KEY"] = "<API-KEY>"
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7)
result = llm.invoke("What are some key concepts of Java Spring Boot?")
print(result.text)
