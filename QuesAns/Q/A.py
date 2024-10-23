from datasets import load_dataset

squad = load_dataset("squad")
cnn_dailymail = load_dataset("cnn_dailymail", "3.0.0")
wiki_qa = load_dataset("wiki_qa")

from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-base')

def preprocess_data(example, mode="qa"):
    if mode == "qa":
        input_text = "question: " + example['question'] + " context: " + example['context']
        target_text = example['answers']['text'][0]
    elif mode == "article":
        input_text = "summarize: " + example['article']
        target_text = example['highlights']
    elif mode == "question_gen":
        input_text = "generate question from: " + example['passage']
        target_text = example['question']
    return {"input_text": input_text, "target_text": target_text}

squad = squad.map(lambda x: preprocess_data(x, mode="qa"), remove_columns=["id"])

cnn_dailymail = cnn_dailymail.map(lambda x: preprocess_data(x, mode="article"))

wiki_qa = wiki_qa.map(lambda x: preprocess_data(x, mode="question_gen"))

from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments

model = T5ForConditionalGeneration.from_pretrained('t5-base')

def tokenize_data(examples):
    model_inputs = tokenizer(examples["input_text"], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["target_text"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

squad_tokenized = squad.map(tokenize_data, batched=True)

cnn_dailymail_tokenized = cnn_dailymail.map(tokenize_data, batched=True)

wiki_qa_tokenized = wiki_qa.map(tokenize_data, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=squad_tokenized["train"],
    eval_dataset=squad_tokenized["validation"],
)

trainer.train()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=cnn_dailymail_tokenized["train"],
    eval_dataset=cnn_dailymail_tokenized["validation"],
)
trainer.train()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=wiki_qa_tokenized["train"],
    eval_dataset=wiki_qa_tokenized["validation"],
)
trainer.train()

def generate_structured_answer(question, context):
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    structured_answer = "\n".join(["- " + sentence.strip() for sentence in answer.split(".") if sentence])
    return structured_answer

context = "Artificial Intelligence (AI) has made significant strides in healthcare, finance, and transportation."
question = "What industries have been impacted by AI?"
print(generate_structured_answer(question, context))

def generate_article(prompt):
    input_text = f"write an article with sections (introduction, body, conclusion) on: {prompt}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(input_ids, max_length=1000, num_beams=4, early_stopping=True)
    article = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return article

prompt = "the future of renewable energy"
print(generate_article(prompt))

def generate_questions_from_passage(passage):
    input_text = f"generate question from: {passage}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return question

passage = "Artificial Intelligence (AI) has made significant strides in healthcare, finance, and transportation."
print(generate_questions_from_passage(passage))

context = "The renewable energy sector is growing rapidly, creating jobs, reducing carbon emissions, and decreasing dependence on fossil fuels."
question = "What are the benefits of renewable energy?"
print(generate_structured_answer(question, context))

prompt = "the role of AI in education"
print(generate_article(prompt))

passage = "AI is revolutionizing education by providing personalized learning experiences and automating administrative tasks."
print(generate_questions_from_passage(passage))

