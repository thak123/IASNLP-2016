# a program to merge the two data sets 

# function to read from text files and returns a list of lines (INPUT: str, RETURN: str)
def text_to_lines( file_name ):
	with open( file_name, 'r') as txt_file:
		text = txt_file.read()

	return text.split('\n')[:-1]

# a function to check whether or not a word exist in our corpus (INPUT: dict, RETURN: boolean)
def isInCorpus( word ):
	if word in word_clusters.keys():
		return True
	else:
		return False

corpus = text_to_lines('sim-scores-adj-200.txt')

word_clusters = {}

for line in corpus:
	words = line.split()

	if words:
		word_clusters[words[0]] = words[1:]

data_old = text_to_lines('training_data_old.txt')
data_new = text_to_lines('training_data_2.txt')
data_els = text_to_lines('else_class.txt')

pairs_old = []
pairs_new = []
pairs_els = []

for line in data_old:
	words = line.split()
	w1, w2 = sorted([words[0], words[1]])
	pairs_old.append([w1,w2,words[2]])

for line in data_new:
	pairs_new.append(line.split())

for line in data_els:
	words = line.split()

	if isInCorpus(words[0]) and isInCorpus(words[1]):
		w1, w2 = sorted([words[0], words[1]])
		pairs_els.append([w1,w2,'ELS'])

final_list = []

for pair in pairs_old + pairs_new + pairs_els:
	if pair not in final_list:
		final_list.append(pair)

count_syn = 0
count_ant = 0
count_els = 0

for point in final_list:
	# print in tsv format
	print point[0] + '\t' + point[1] + '\t' + point[2]
	if point[2] == 'SYN':
		count_syn += 1
	elif point[2] == 'ANT':
		count_ant += 1
	else:
		count_els += 1

# print len(final_list)
# print count_syn
# print count_ant
# print count_els
