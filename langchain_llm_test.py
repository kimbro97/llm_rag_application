from dotenv import load_dotenv
load_dotenv()

from langchain_upstage import ChatUpstage

llm = ChatUpstage()

ai_message = llm.invoke("인프런에 어떤 강의가 있나여? ")

print(ai_message)


