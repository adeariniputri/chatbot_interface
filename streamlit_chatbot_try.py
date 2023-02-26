import nltk
nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy
import tflearn
from tensorflow.python.framework import ops
import random
import json
#from tflearn.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
import jsonify
import streamlit as st
from streamlit_chat import message


ops.reset_default_graph()
net = tflearn.input_data(shape=[None, 64])
net = tflearn.fully_connected(net, 32)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 16)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8, activation="softmax")
adam = Adam(learning_rate=0.001, beta1=0.99)
net = tflearn.regression(net, optimizer=adam)
model = tflearn.DNN(net)
model.load('saved_model/model.tflearn')
print(model)

with open('intents.json') as file:
	data = json.load(file)

labels = []
words = []
for i in data["knowledge"]:
    if i["tag"] not in labels:
        labels.append(i["tag"])
    for j in i["patterns"]:
        word = nltk.word_tokenize(j)
        words.extend(word)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

def generateArrays(s, n):
	arrVals = [0 for _ in range(len(n))]
	s_words = nltk.word_tokenize(s)
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(n):
			if w == se:
				arrVals[i] = 1

	return numpy.array(arrVals)

# console
def generateAutoChat(inp):
    result = model.predict([generateArrays(inp,words)])[0]
    result_index = numpy.argmax(result)
    tag = labels[result_index]
    	
    if result[result_index] > 0.75:
        for tg in data["knowledge"]:
            if tg["tag"] == tag:
                responses = tg['responses']
                return random.choice(responses)
    else:
        return "try"
    		


st.title("Ade's bot")

# Add an input field for the user to enter messages
user_input = st.text_input("Enter your message:")

# Add a button to send the message to the chatbot
if st.button("Send"):
    message_history_bot = []
    bot_response = generateAutoChat(user_input)
    message_history_bot.append(bot_response)
    for message_ in message_history_bot:
        message(user_input, is_user=True) 
        message(message_)   