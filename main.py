import fastapi
from chromaModel import ChromaSetup

app = fastapi.FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "Welcome to car and general information API!"}


@app.get("/assistant")
def assistant():
    return {
        "chat is a smart assistant that can answer questions about car and general information.about car and general from the document provided to it "
    }


@app.get("/chat")
def chat(request: str):
    print(f"Received request: {request}")
    response = ChromaSetup(request)
    return {"response": response}
