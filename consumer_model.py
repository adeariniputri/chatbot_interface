#Import library yang dibutuhkan
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy
import tflearn
import tensorflow
from tensorflow.python.framework import ops
import random
import json
import pickle

#Membuka file (Jadikan dalam satu folder file json dengan file python main)
with open(r'D:\jobs\chatbot\intents.json') as file:
	data = json.load(file)
#Menggunakan try dan except untuk menghindari training data dua kali.
try:
	with open("data.pickle", "rb") as f:
		kata, label, training, output = pickle.load(f)
except:

	kata = []
	label = []
	dokumen_x = []
	dokumen_y = []
	#Proses tokenizing
	for intens in data["intents"]:
		for pattern in intens["patterns"]:
			kta = nltk.word_tokenize(pattern)
			kata.extend(kta)
			dokumen_x.append(kta)
			dokumen_y.append(intens['tag'])

			if intens["tag"] not in label:
				label.append(intens["tag"])
	#Proses stemming dengan library sastrawi
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	kata = [stemmer.stem(w.lower()) for w in kata if w != "?"]
	kata = sorted(list(set(kata)))
	#Mengurutkan hasil stemming
	label = sorted(label)
	#Persiapan modeling
	training = []
	output = []

	out_empty = [0 for _ in range(len(label))]

	for x, doc in enumerate(dokumen_x):
		bag = []

		kta = [stemmer.stem(w) for w in doc]

		for w in kata:
			if w in kta:
				bag.append(1)
			else:
				bag.append(0)

		output_row = out_empty[:]
		output_row[label.index(dokumen_y[x])] = 1

		training.append(bag)
		output.append(output_row)

	training = numpy.array(training)
	output = numpy.array(output)

	with open("data.pickle", "wb") as f:
		pickle.dump((kata, label, training, output),f)

#Proses pembuatan model
ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.load("saved_model/model.tflearn")

def bag_of_words(s, kata):
	bag = [0 for _ in range(len(kata))]

	s_words = nltk.word_tokenize(s)
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(kata):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)

def autoChat(inp):
	result = model.predict([bag_of_words(inp,kata)])[0]
	result_index = numpy.argmax(result)
	tag = label[result_index]
	if result[result_index] > 0.80:
		for tg in data["intents"]:
			if tg["tag"] == tag:
				responses = tg['responses']
		return random.choice(responses),result[result_index]
	else:
		return "Maaf, saya tidak mengerti yang Anda tanyakan",result[result_index]
