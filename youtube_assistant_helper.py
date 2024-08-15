from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.llms.openai import OpenAI
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


#from langchain.chains import LLMChain

from langchain_core.runnables.base import RunnableSequence
from langchain_community.vectorstores.faiss import FAISS
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

embeddings = OpenAIEmbeddings()

def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    docs = text_splitter.split_documents(transcript)
    
    db = FAISS.from_documents(docs, embeddings)
    return db
    #return docs #to see text

#print(create_vector_db_from_youtube_url(video_url))

def get_response_from_query(db, query, k = 4):
    # text davinci can handle 4097 tokens
    docs = db.similarity_search(query, k = k)
    docs_page_content = " ".join([d.page_content for d in docs])
    llm = ChatOpenAI(model = "gpt-4o")   
    prompt = PromptTemplate(
        input_variable = ["question", "docs"],
        template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": query, "docs": docs_page_content})
    return answer
    
    # sequence = RunnableSequence(llm = llm, prompt = prompt)
    # response = sequence.run(question = query, docs = docs_page_content) # problema ir seit jo tutorial dzeks lietoja LLMChain, bet sobrid tas ir deprecated un suggesto lietot runnablesequence
    # response = response.replace("\n", "")
    
    # return response
    
if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=1bUy-1hGZpI"
    question = "What is the video about?"
    db = create_vector_db_from_youtube_url(video_url)
    response = get_response_from_query(db, question)
    print(response)