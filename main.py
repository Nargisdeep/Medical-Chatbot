from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertForQuestionAnswering
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("fine_tuned_intent_model")
intent_model = DistilBertForSequenceClassification.from_pretrained("fine_tuned_intent_model")
qa_model = DistilBertForQuestionAnswering.from_pretrained("fine_tuned_qa_model")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load symptom data
with open("symptom_data.json", "r") as f:
    symptom_data = json.load(f)

# Prepare FAISS index for vector search
responses = [intent["response"] for intent in symptom_data["intents"]]
response_embeddings = embedder.encode(responses)
dimension = response_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(response_embeddings)

# Intent labels
intent_labels = ["greeting", "small_talk", "symptom_query"]

class Query(BaseModel):
    text: str

def classify_intent(query: str) -> str:
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = intent_model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    logger.info(f"Query: {query}, Predicted class: {predicted_class}, Intent: {intent_labels[predicted_class] if predicted_class < len(intent_labels) else 'out_of_scope'}")
    return intent_labels[predicted_class] if predicted_class < len(intent_labels) else "out_of_scope"

def get_response(query: str, intent: str) -> str:
    if intent == "out_of_scope":
        return "I'm sorry, I'm only trained to answer questions about abdominal pain in adults. Could you ask something related to that?"
    
    query_embedding = embedder.encode([query])[0]
    _, indices = index.search(np.array([query_embedding]), k=1)
    matched_response = responses[indices[0][0]]
    
    if intent == "symptom_query":
        inputs = tokenizer(query, matched_response, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = qa_model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        # Ensure the span is valid and excludes special tokens
        if start < end and start > 0 and end < inputs["input_ids"].shape[1]:  # Avoid [CLS] and [SEP]
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end]))
            logger.info(f"QA Answer: {answer}, Start: {start}, End: {end}")
            if answer.strip() and not answer.startswith("[") and not answer.endswith("]"):
                return answer
        logger.warning(f"QA model returned invalid or empty answer, falling back to matched response")
        return matched_response
    return matched_response

@app.post("/chat")
async def chat(query: Query):
    try:
        intent = classify_intent(query.text)
        response = get_response(query.text, intent)
        return {"response": response, "intent": intent}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))