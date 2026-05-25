<h1 align="center"> Natural Language Processing  with Hugging Face Transformers </h1>
<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

## Name : Felis Eren Cristi Milala

## My todo : 

### 1. Example 1 - Sentiment Analysis

```
# TODO :
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

classifier("I really like watching the sunset because it makes me feel calm.")
```
Result : 

```
[{'label': 'POSITIVE', 'score': 0.9985151886940002}]
```

Analysis on example 1 : 

The sentiment analysis model successfully identifies the sentence as positive. The sentence expresses a happy and calm feeling, so the positive label is appropriate. The high confidence score shows that the model is quite sure about its prediction.


### 2. Example 2 - Topic Classification

```
# TODO :
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

classifier(
    "Artificial intelligence is helping people build smarter applications and automate daily tasks.",
    candidate_labels=["technology", "sports", "politics", "health"]
)
```

Result : 

```
{'sequence': 'Artificial intelligence is helping people build smarter applications and automate daily tasks.',
 'labels': ['technology', 'health', 'sports', 'politics'],
 'scores': [0.9904135465621948,
  0.005775279365479946,
  0.002484129276126623,
  0.0013270135968923569]}
```

Analysis on example 2 : 

The zero-shot classification model is able to classify the sentence into the most relevant topic. Since the sentence talks about artificial intelligence, smart applications, and automation, the most suitable label is technology. This shows that the model can classify text into categories even without specific training on those labels.

### 3. Example 3 and 3.5 - Text Generator

```
# TODO :
generator = pipeline("text-generation", model="distilgpt2")
generator(
    "Artificial intelligence is rapidly changing the world, and learning how to code will open many doors for",
    max_length=30,
    num_return_sequences=2,
)
```

Result : 

```
[{'generated_text': 'Artificial intelligence is rapidly changing the world, and learning how to code will open many doors for the future. We are just getting started.'},
 {'generated_text': 'Artificial intelligence is rapidly changing the world, and learning how to code will open many doors for future AI experiments.\n\n\n\n\nThere are'}]
```

Analysis on example 3 : 

The text generation model continues the given sentence by generating new text based on the input prompt. The result may be different each time because the model generates text creatively. This task shows how transformer models can be used to produce human-like text.

```
unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("The sky is very <mask> today, so I think it might rain later.", top_k=4)
```

Result : 

```
[{'score': 0.6047289967536926,
  'token': 2779,
  'token_str': ' cloudy',
  'sequence': 'The sky is very cloudy today, so I think it might rain later.'},
 {'score': 0.08139749616384506,
  'token': 4520,
  'token_str': ' bright',
  'sequence': 'The sky is very bright today, so I think it might rain later.'},
 {'score': 0.06473786383867264,
  'token': 699,
  'token_str': ' clear',
  'sequence': 'The sky is very clear today, so I think it might rain later.'},
 {'score': 0.03231789916753769,
  'token': 10521,
  'token_str': ' grey',
  'sequence': 'The sky is very grey today, so I think it might rain later.'}]
```

Analysis on example 3.5 : 

The fill-mask model predicts the most suitable word to replace the masked token. In this sentence, words related to weather such as cloudy or dark are appropriate because the sentence mentions that it might rain later. This shows that the model understands the context of the sentence.

### 4. Example 4 - Name Entity Recognition (NER)

```
# TODO :
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
ner("My name is Felis, I am a student in Universitas Diponegoro, and I am learning NLP with Hugging Face.")
```

Result : 

```
[{'entity_group': 'PER',
  'score': np.float32(0.9986745),
  'word': 'Felis',
  'start': 11,
  'end': 16},
 {'entity_group': 'ORG',
  'score': np.float32(0.90793145),
  'word': 'Universitas Diponegoro',
  'start': 36,
  'end': 58},
 {'entity_group': 'MISC',
  'score': np.float32(0.9450888),
  'word': 'NL',
  'start': 78,
  'end': 80},
 {'entity_group': 'PER',
  'score': np.float32(0.9151961),
  'word': 'Face',
  'start': 95,
  'end': 99}]
```

Analysis on example 4 : 

The Named Entity Recognition model identifies important entities in the sentence. It detects Diaz as a person, Indonesia as a location, and Hugging Face as an organization. This shows that NER can be used to extract names, places, and organizations from text.

### 5. Example 5 - Question Answering

```
# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "Where is the Eiffel Tower located?"
context = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed in 1889."
qa_model(question=question, context=context)
```

Result : 

```
{'score': 0.35826337337493896,
 'start': 56,
 'end': 86,
 'answer': 'Champ de Mars in Paris, France'}
```

Analysis on example 5 : 

The question-answering model successfully extracts the location of the Eiffel Tower from the given context. The answer returned is "Champ de Mars in Paris, France", which is more specific than just "Paris". This shows that the model can find and return the most relevant answer span from the text instead of generating a new answer.

### 6. Example 6 - Text Summarization

```
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

text_to_summarize = """
Felis is a student who is currently learning natural language processing using Hugging Face Transformers.
This topic is important because it helps computers understand, analyze, and generate human language.
By studying NLP, Diaz can learn how to build useful applications such as sentiment analysis tools,
question answering systems, text summarizers, translation programs, and named entity recognition models.
These technologies are widely used in education, business, healthcare, and many digital platforms.
Learning Hugging Face is also useful because it provides simple pipelines and pre-trained models that make
machine learning projects easier to build and test. With enough practice, Diaz can improve programming skills
and understand how artificial intelligence is applied in real-world problems.
"""
summarizer(text_to_summarize)
```

Result : 

```
[{'summary_text': ' Felis Diaz is currently learning natural language processing using Hugging Face Transformers . NLP helps computers understand, analyze, and generate human language . Diaz can learn how to build useful applications such as sentiment analysis tools, question answering systems, text summarizers, translation programs, and named entity recognition models .'}]

```

Analysis on example 6 :

The summarization model shortens the original paragraph while keeping the main idea. The summary still explains that Felis is learning NLP with Hugging Face Transformers and mentions several NLP applications. This shows that summarization can help reduce long text into a shorter and meaningful version.

### 7. Example 7 - Translation

```
# TODO :
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Saya sedang belajar teknologi kecerdasan buatan menggunakan Hugging Face.")
```

Result : 

```
[{'translation_text': "J'étudie la technologie artificielle de l'intelligence en utilisant la Face Hugging."}]

```

Analysis on example 7 :

The translation model translates the Indonesian sentence into French. The result keeps the original meaning, which is about learning artificial intelligence technology using Hugging Face Transformers. This shows that transformer models can be used for multilingual communication.

---

## Analysis on this project

This project offers a practical introduction to various NLP tasks using Hugging Face pipelines. Each example is easy to follow and demonstrates real-world use cases. The variety of models shows the flexibility of transformer-based solutions in solving different types of language problems.
