from analytics import AnalyticsTracker
tracker = AnalyticsTracker()
import random
import json
import pickle
import numpy as np
import tensorflow as tf 
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import threading
import dashboard
from dashboard import run_dashboard
dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
dashboard_thread.start()

lemmatizer = WordNetLemmatizer()
def get_response(user_input):
    
    intent = "detected_intent"  
    response = "generated_response"  
    
    
    tracker.log_interaction(
        user_id="current_user",  
        query=user_input,
        intent=intent,
        response=response
    )
    
    return response

intents = json.load(open(r'E:\chatbot\data\intents.json'))
                        
words =[]
classes = []
documents = []
ignoreLetters = ["?","!",".",","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent["tag"]))
        if intent ["tag"] not in classes:
            classes.append(intent["tag"])
            
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open("words.pkl","wb"))
pickle.dump(classes, open("classes.pkl","wb"))

training = []
outputEmpty = [0]*len(classes)

for documents in documents:
    bag = []
    wordPatterns = documents[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)
        
    outputRow = list(outputEmpty)
    outputRow[classes.index(documents[1])] = 1
    training.append(bag + outputRow)
    
random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]
 
if __name__ == "__main__":
    # Test interaction
    test_input = "Hello chatbot!"
    print("Chatbot Response:", get_response(test_input))
    print("Interaction logged in analytics!")
    
def log_interaction(user_query, bot_response):
    log_data = {
        "timestamp": str(datetime.datetime.now()),
        "query": user_query,
        "response": bot_response
    }

    # Append to a JSON file
    with open("chat_logs.json", "a") as logfile:
        logfile.write(json.dumps(log_data) + "\n")

    print(f"Interaction Logged: {log_data}")  # Debugging statement


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128,input_shape = (len(trainX[0]),), activation = "relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation ="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(classes), activation = "softmax"))

sgd = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9, nesterov = True)
model.compile(loss="categorical_crossentropy", optimizer=sgd , metrics = ["accuracy"])

hist = model.fit(
    np.array(trainX), np.array(trainY), epochs=400, batch_size=5, verbose=1
)
model.save("chatbot_model.h5", hist)
print("done")