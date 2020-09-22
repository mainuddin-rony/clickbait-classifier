import fasttext

CLICKBAIT_CLASSIFIER = fasttext.load_model('classification_model.bin')

while True:
	headline = raw_input("Enter headline : ")
	link_name = headline.lower() 
	lines = [link_name]
	labels = CLICKBAIT_CLASSIFIER.predict_proba(lines)
	print('Decision: ', str(labels[0][0][0]))
	print('Probability: ', labels[0][0][1])



