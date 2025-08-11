import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from torch.utils.data import Dataset
import json
from sklearn.model_selection import train_test_split

# Initialize tokenizer and models
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
intent_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
qa_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
intent_labels = ["greeting", "small_talk", "symptom_query"]

# Custom Dataset for Intent Classification
class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Custom Dataset for Question Answering
class QADataset(Dataset):
    def __init__(self, questions, contexts, start_positions, end_positions):
        self.questions = questions
        self.contexts = contexts
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.encodings = tokenizer(questions, contexts, padding=True, truncation=True, return_tensors="pt")

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["start_positions"] = torch.tensor(self.start_positions[idx])
        item["end_positions"] = torch.tensor(self.end_positions[idx])
        return item

# Load and prepare data
def load_data():
    with open("symptom_data.json", "r") as f:
        data = json.load(f)

    # Intent classification data
    intent_texts = []
    intent_labels_idx = []
    for intent in data["intents"]:
        for example in intent["examples"]:
            intent_texts.append(example)
            intent_labels_idx.append(intent_labels.index(intent["type"]))

    # Question-answering data
    qa_questions = []
    qa_contexts = []
    qa_start_positions = []
    qa_end_positions = []
    for intent in data["intents"]:
        if intent["type"] == "symptom_query":
            context = intent["response"]
            for example in intent["examples"]:
                qa_questions.append(example)
                qa_contexts.append(context)
                tokens = tokenizer.encode(context, add_special_tokens=False)
                answer_start = 0
                answer_end = len(tokens) - 1
                qa_start_positions.append(answer_start)
                qa_end_positions.append(answer_end)

    return intent_texts, intent_labels_idx, qa_questions, qa_contexts, qa_start_positions, qa_end_positions

# Fine-tuning function
def fine_tune_models():
    intent_texts, intent_labels_idx, qa_questions, qa_contexts, qa_start_positions, qa_end_positions = load_data()

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        intent_texts, intent_labels_idx, test_size=0.2, random_state=42
    )
    qa_train_indices, qa_val_indices = train_test_split(
        range(len(qa_questions)), test_size=0.2, random_state=42
    )
    train_qa_questions = [qa_questions[i] for i in qa_train_indices]
    val_qa_questions = [qa_questions[i] for i in qa_val_indices]
    train_qa_contexts = [qa_contexts[i] for i in qa_train_indices]
    val_qa_contexts = [qa_contexts[i] for i in qa_val_indices]
    train_qa_start_positions = [qa_start_positions[i] for i in qa_train_indices]
    val_qa_start_positions = [qa_start_positions[i] for i in qa_val_indices]
    train_qa_end_positions = [qa_end_positions[i] for i in qa_train_indices]
    val_qa_end_positions = [qa_end_positions[i] for i in qa_val_indices]

    # Create datasets
    train_intent_dataset = IntentDataset(train_texts, train_labels)
    val_intent_dataset = IntentDataset(val_texts, val_labels)
    train_qa_dataset = QADataset(train_qa_questions, train_qa_contexts, train_qa_start_positions, train_qa_end_positions)
    val_qa_dataset = QADataset(val_qa_questions, val_qa_contexts, val_qa_start_positions, val_qa_end_positions)

    # Training arguments for intent model
    intent_training_args = TrainingArguments(
        output_dir="./intent_results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./intent_logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Trainer for intent model
    intent_trainer = Trainer(
        model=intent_model,
        args=intent_training_args,
        train_dataset=train_intent_dataset,
        eval_dataset=val_intent_dataset,
    )

    # Train intent model
    intent_trainer.train()
    intent_model.save_pretrained("fine_tuned_intent_model")
    tokenizer.save_pretrained("fine_tuned_intent_model")

    # Training arguments for QA model
    qa_training_args = TrainingArguments(
        output_dir="./qa_results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./qa_logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Trainer for QA model
    qa_trainer = Trainer(
        model=qa_model,
        args=qa_training_args,
        train_dataset=train_qa_dataset,
        eval_dataset=val_qa_dataset,
    )

    # Train QA model
    qa_trainer.train()
    qa_model.save_pretrained("fine_tuned_qa_model")
    tokenizer.save_pretrained("fine_tuned_qa_model")

if __name__ == "__main__":
    fine_tune_models()


# // {
#   //   "type": "out_of_scope",
#   //   "examples": ["Tell me about heart disease", "What’s the weather today?", "Explain machine learning", "Can you tell me about diabetes?"],
#   //   "response": "I'm currently trained to help with abdominal pain in adults only. I can’t provide details on that topic."
#   // },

#   //   {
#   //     "type": "out_of_scope",
#   //     "examples": ["What causes headaches?", "How to treat a cold?"],
#   //     "response": "I'm sorry, I'm only trained to answer questions about abdominal pain in adults. Could you ask something related to that?"
#   //   }