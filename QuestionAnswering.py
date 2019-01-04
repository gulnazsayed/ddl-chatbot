# WILL BE ADDED TO SUMMARIZETEXT UPON COMPLETION

# function to parse a given sentence
def parse(sentence):
    nlp = StanfordCoreNLP(path_or_host='http://localhost', port=9000, timeout=30000)
    props = {'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation', \
             'pipelineLanguage': 'en', 'outputFormat': 'json'}
    parse = nlp.parse(sentence)
    return parse

#function to parse and get the Named entities from a given sentence
def parse_ner(sentence):
    nlp = StanfordCoreNLP(path_or_host='http://localhost',port=9000,timeout=30000)
    props = {'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',\
                  'pipelineLanguage': 'en','outputFormat': 'json'}
    parse = nlp.ner(sentence)
    return parse

#function to find out the question type
def get_label(question):
    tree = parse(question)
    tree = Tree.fromstring(str(tree))
    for s in tree.subtrees(lambda t: t.label() == "WP" or t.label() == "WRB"):
        if "who" in s.leaves() or "Who" in s.leaves():
            return "who"
        if "what" in s.leaves() or "What" in s.leaves():
            return "what"
        if "why" in s.leaves() or "Why" in s.leaves():
            return "why"
        if "where" in s.leaves() or "Where" in s.leaves():
            return "where"
        if "when" in s.leaves() or "When" in s.leaves():
            return "when"
        else:
            return "other"
    for s in tree.subtrees(lambda t: t.label() == "WP$"):
        return "other"
    return "binary"

#cosine similarity functions to extract relevant sentence among sentences based on question
def cleaning_text(text):
    cleaned_text = []
    for i in range(0,len(text)):
        #remove punctuation
        words = re.sub('[^a-zA-Z]', ' ', (text[i]))
        #convert all to lowercase
        words = words.lower()
        words = words.split()
        #stemming
        ps = PorterStemmer()
        words = [ps.stem(word) for word in words if not word in set(stopwords.words('english'))]
        words = ' '.join(words)
        cleaned_text.append(words)
    return cleaned_text

def cleaning_query(query):
    words = re.sub('[^a-zA-Z]', ' ', query)
    #convert all to lowercase
    words = words.lower()
    words = words.split()
    #stemming
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if not word in set(stopwords.words('english'))]
    words = ' '.join(words)
    return words

def find_sentence(question, sentences):
    cleaned_sentences = cleaning_text(sentences)
    cleaned_question = cleaning_query(question)
    cleaned_sentences.append(cleaned_question)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_sentences)
    output_array=cosine_similarity(tfidf_matrix[-1], tfidf_matrix)
    output_array=np.transpose(output_array)

    index, value = max(enumerate(output_array[:-1]), key=operator.itemgetter(1))
    answer = sentences[index]
    return answer


# function to find answer when question is of type 'why'
def why_answer(question, relevant):
    r_parsed = parse(relevant)
    r_tree = Tree.fromstring(r_parsed)
    answer = []
    for s in r_tree.subtrees(lambda t: t.label() == "SBAR"):
        answer = s.leaves() if s[0].label() == "IN" else []
        break

    if len(answer) > 0:
        answer = untokenize(answer)
    else:
        return relevant
    answer = answer[0].upper() + answer[1:] + "."
    return answer

#function to find answer when the question is of type 'where' or 'when'
def get_answer_when_or_where(question, relevant):
    r_parsed = parse(relevant)
    r_tree = Tree.fromstring(r_parsed)
    for i in range(len(r_tree[0])):
        node = r_tree[0][i]
        if i == 0 and node.label() == "PP" and " ".join(node.leaves()).lower() not in question.lower():
            answer = " ".join(node.leaves()) + "."
            answer = answer[0].upper() + answer[1:]
            return answer
        if node.label() == "VP":
            for sub_node in node:
                if (sub_node.label() == "PP" or sub_node.label() == "SBAR") and " ".join(sub_node.leaves()).lower() not in question.lower():
                    answer = " ".join(sub_node.leaves()) + "."
                    answer = answer[0].upper() + answer[1:]
                    return answer
    return relevant

## functions to finding out answer to binary question
BE_VB_LIST = ["is", "was", "are", "am", "were", "will", "would", "could", "might", "may", "should", "can"]
DO_DID_DOES = ["do", "did", "does"]
VB_LIST = ["VBZ", "VBP", "VBD"]

def bin_answer(question, relevant):
    if get_binary_answer(question,relevant):
        if positive_bin_answer(relevant):
            return 'Yes.'
        else:
            return 'No.'
    else:
        return relevant


def get_binary_answer(question, answer):
    question = question[0].lower() + question[1:]
    q_tree = parse(question)
    q_tree = Tree.fromstring(str(q_tree))
    a_tree = parse(Binary_main(answer))
    a_tree = Tree.fromstring(str(a_tree))
    res = True
    (q_top_level_structure, q_parse_by_structure) = Answer_Bin_get_top_level_structure(q_tree)
    (a_top_level_structure, a_parse_by_structure) = Answer_Bin_get_top_level_structure(a_tree)
    q_parse_by_structure[0][0] = q_parse_by_structure[0][0].lower()
    for i in range(0, len(q_top_level_structure)):
        q_label = q_top_level_structure[i]
        if q_label in a_top_level_structure:
            a_index = a_top_level_structure.index(q_label)
        else:

            return False
    if not q_parse_by_structure == a_parse_by_structure:
        return False
    return True


def Answer_Bin_get_top_level_structure(tree):
    top_level_structure = []
    parse_by_structure = []
    for t in tree.subtrees(lambda t: t.label() == "SQ"):
        for tt in t:
            top_level_structure.append(tt.label())
            parse_by_structure.append(tt.leaves())
        return (top_level_structure, parse_by_structure)


def Binary_main(text):
    tree = parse(text)
    tree = Tree.fromstring(str(tree))
    (sent, NEG, is_binary) = convert(text, tree)
    if not is_binary:
        return False
    tree = parse(sent)
    tree = Tree.fromstring(str(tree))

    return bin_q_type(tree, sent, NEG)

def convert(text, tree):
    parse_by_structure = []
    NEG = 0
    NP = 0
    VP = 0
    for t in tree[0]:
        if t.label() == "VP":
            VP = 1
        if t.label() == "NP":
            NP = 1
        if t.label() != "VP":
            parse_by_structure += (t.leaves())
        else:
            for tt in t:
                if tt.label() != "RB":
                    parse_by_structure += (tt.leaves())
                else:
                    NEG = 1
    sent = " ".join(parse_by_structure)
    is_binary = NP and VP
    return (sent, NEG, is_binary)

def bin_q_type(tree, text, neg):
    (top_level_structure, parse_by_structure) = get_top_level_structure(tree)
    verb_index = top_level_structure.index("VB")
    verb = parse_by_structure[verb_index]
    np_index = top_level_structure.index("NP")
    if verb in BE_VB_LIST or neg == 1:
        return be_q(parse_by_structure, verb_index, np_index)
    else:
        return do_q(parse_by_structure, verb_index, np_index)

def get_top_level_structure(tree):
    top_level_structure = []
    parse_by_structure = []
    for t in tree[0]:
        if t.label() == "VP":
            parse_by_structure += get_VB(t)
            top_level_structure += ["VB", "OTHER"]
        else:
            parse_by_structure.append(" ".join(t.leaves()))
            top_level_structure.append(t.label())
    return (top_level_structure, parse_by_structure)

def get_VB(subtree):
    res = []
    for t in subtree:
        if t.label() in VB_LIST:
            res.append(" ".join(t.leaves()))
        else:
            res.append(" ".join(t.leaves()))
    return (res)

def be_q(parse_by_structure, verb_index, np_index):
    verb = parse_by_structure[verb_index]
    nb = parse_by_structure[np_index]
    sent = parse_by_structure
    sent[verb_index] = nb
    sent[np_index] = verb
    sent[-1] = "?"
    sent = " ".join(sent)
    #sent = sent[0].upper() + sent[1:]
    return sent


def do_q(parse_by_structure, verb_index, np_index):
    verb = parse_by_structure[verb_index]
    (tense, person, a, b, c) = tenses(verb)[0]
    present_verb = str(conjugate(verb, tense="present", person=1))
    sent = parse_by_structure
    sent[verb_index] = present_verb
    if tense == 'past':
        sent.insert(np_index, "did")
    elif tense == 'present' and person == 3:
        sent.insert(np_index, "does")
    else:
        sent.insert(np_index, "do")
    sent[-1] = "?"
    sent = " ".join(sent)
    # sent = sent[0].upper() + sent[1:]
    return sent

## see if there is a 'not' in the answer
def positive_bin_answer(answer):
    tree = parse(answer)
    tree = Tree.fromstring(str(tree))
    for t in tree[0]:
        if t.label() == "VP":
            VP = 1
            for tt in t:
                if tt.label() == "RB":
                    return False
            return True

##functions to find answers when question is of type 'who'
def get_q_noun(question):
    tree = parse(question)
    tree = Tree.fromstring(str(tree))
    for s in tree.subtrees(lambda t: t.label() == "NP" or t.label() == "NNP" or t.label() == "NN"):
                    for ss in s.subtrees(lambda t: t.label() == "NN" or t.label() == "NNS" or t.label()=='NNP'):
                        return ss.leaves() ## should return only the nouns
def get_q_verb(question):
    tree = parse(question)
    tree = Tree.fromstring(str(tree))
    for s in tree.subtrees(lambda t: t.label() == "VBZ" or t.label() == "VB" or t.label() =="VBD"):
                    return s.leaves()
def get_who_answer(question,relevant,svos):
    answers = []
    answers_str=''
    q_noun = get_q_noun(question)[0] if get_q_noun(question) is not None else None
    q_verb = get_q_verb(question)[0] if get_q_verb(question) is not None else None
    if q_noun is None or q_verb is None:
        return relevant
    for sentence in svos:
        for tuple in sentence:
            subj = tuple[0]
            verb = tuple[1]
            obj = tuple[2]
            # lemmatise both question verb and verb
            verb = wordnet_lemmatizer.lemmatize(verb.lower(), pos='v')
            q_verb = wordnet_lemmatizer.lemmatize(q_verb.lower(), pos='v')
            if q_verb.lower()==verb.lower():
            # if the name is the subject, then answer is the object
            #check if the name in question and answer are same, partial name match also..
                if re.search(q_noun.lower(),subj.lower().replace('(','').replace(')','')) \
                or re.search(subj.lower().replace('(','').replace(')',''), q_noun.lower()):
                    answers.append(obj)
                elif re.search(q_noun.lower(),obj.lower().replace('(','').replace(')','')) \
                or re.search(obj.lower().replace('(','').replace(')',''), q_noun.lower()):
                    answers.append(subj)
    if len(answers) > 0:
        for answer in answers:
            answers_str = answers_str + answer + ','

        answers_str = answers_str[:-1]
        answers_str = answers_str + '.'
        return answers_str
    else:
        return relevant

# function for asking questions and getting answers
def ask_question(corpus,svos):
    while 1:
        print()
        question = input("Ask your question, or press q to quit\n")
        if question=='q':
            break
        else:
            label = get_label(question)
            print()
            print('Question:',question)
            #extract relevant sentence from passage based on question
            relevant = find_sentence(question,corpus)
            if label == "when":
                print('Answer:',get_answer_when_or_where(question, relevant))
            elif label == "where":
                print('Answer:',get_answer_when_or_where(question, relevant))
            elif label=='binary':
                print('Answer:',bin_answer(question,relevant))
            elif label=='who':
                print('Answer:',get_who_answer(question,relevant,svos))
            else:
                print('Answer:',relevant)

# GENERATING QUESTIONS FROM THE TEXT
# Functions for asking why questions
def is_why(tree):
    for t in tree.subtrees(lambda t: t.label() == "SBAR"):
        if "because" in t.leaves() or "since" in t.leaves() or "so" in t.leaves():
            return True
    return False


def remove_SBAR(tree):
    top_level_structure = []
    parse_by_structure = []
    for t in tree[0]:
        if t.label() != "SBAR" and t.label() != "VP" and t.label() != ",":
            parse_by_structure.append(" ".join(t.leaves()))
            top_level_structure.append(t.label())
        elif t.label() == "VP":
            for tt in t:
                if tt.label() != "SBAR":
                    parse_by_structure.append(" ".join(tt.leaves()))
                    top_level_structure.append(tt.label())
    return (top_level_structure, parse_by_structure)


def why_q_main(text):
    # print(text)
    tree = parse(text)
    tree = Tree.fromstring(str(tree))
    # print tree
    if not is_why(tree):
        return None  # print ("It could not be converted to why question.")
    (top_level_structure, parse_by_structure) = remove_SBAR(tree)
    # print top_level_structure
    # print parse_by_structure
    sent = " ".join(parse_by_structure)
    sent = Binary_main(sent)
    print("Why " + sent)
    return "Why " + sent

#### Functions for asking who question
def is_who(text, NE):
    for ne in NE:
        if ne in text:
            return True
    tt = ne_chunk(pos_tag(word_tokenize(text)))
    for s in tt.subtrees(lambda t: t.label() == "PERSON"):
        return True
    return False


def who_main(text, NE):
    tree = parse(text)
    tree = Tree.fromstring(str(tree))
    (top_level_structure, parse_by_structure) = get_top_level_structure(tree)
    np_index = top_level_structure.index("NP") if "NP" in top_level_structure else None
    if np_index is None:
        return None
    if is_who(parse_by_structure[np_index], NE):
        parse_by_structure[np_index] = "who"
    else:
        return None
    parse_by_structure[-1] = "?"
    sent = " ".join(parse_by_structure)
    sent = sent[0].upper()+sent[1:]
    #print(sent)
    return sent


# When questions
def when_check(node):
    """ check if time related PP """
    if (node.label() == "PP"):
        node_ner = parse_ner(" ".join(node.leaves()))
        time_set = set(["DATE", "TIME"])
        if any(t in time_set for t in reduce(operator.concat, node_ner)):
            return True
        node_leaves = (" ".join(node.leaves())).lower()
        if re.search(r'in (.*?) time', node_leaves, re.M | re.I):
            return True
    return False


def contain_loc(s):
    """ check if s contains prepositions of place and direction """

    loc_set = set(["above", "across", "after",
                   "against", "along", "among",
                   "around", "at", "behind", "below",
                   "beside", "between", "by", "close to",
                   "down", "from", "in front of", "inside",
                   "in", "into", "near", "next to", "off", "on",
                   "onto", "opposite", "out of", "outside", "over",
                   "past", "through"])

    tree = Tree.fromstring(parse(s))
    for i in tree[0]:
        if (str(i.label()) == "PP" and
                any(str(j).lower() in loc_set for j in i.leaves())):
            return "PP"
        if (str(i.label())) == "VP":
            for j in i:
                if (str(j.label()) == "PP" and
                        any(str(k).lower() in loc_set for k in j.leaves())):
                    return "VP"
    return None

def PP_location_deletion(si, C):
    acc = ""
    tree = Tree.fromstring(parse(si))
    if C == "PP":
        for node in tree[0]:
            if node.label() == "PP" or node.label() == ",":
                acc += ""
            else:
                acc += " ".join(node.leaves())
            acc += " "
        return acc
    elif C == "VP":
        for node in tree[0]:
            if node.label() == "VP":
                for sub_node in node:
                    if sub_node.label() == "PP":
                        acc += ""
                    else:
                        acc += " ".join(sub_node.leaves())
                    acc += " "
            else:
                acc += " ".join(node.leaves())
            acc += " "
        return acc
    else:
        return None


def check_when(s):
    tree = Tree.fromstring(parse(s))
    for i in tree[0]:
        if (str(i.label()) == "PP" and (when_check(i))):
            return "when"
        elif (str(i.label())) == "VP":
            for j in i:
                if (str(j.label()) == "PP" and (when_check(j))):
                    return "when"
    return "where"


def when_q(si):
    C = contain_loc(si)
    if C:
        si_filtered = PP_location_deletion(si, C)
        binary_q = Binary_main(si_filtered)
        binary_q = binary_q[0].lower() + binary_q[1:]
        si_binary = binary_q
        if si_binary:
            if check_when(si) == "when":
                return "When " + si_binary
            else:
                return None

    return None


# ##### functions to automatically read from corpus(list of sentences) and create a list of questions...
def top_k_sentences(k, corpus, topK=True):
    acc = []
    if topK:
        # get top k shortest sentences
        sentences_top_k = sorted(corpus, key=len)[:k]
        return sentences_top_k
    else:
        return corpus


# function to test what questions are being generated
def questions_main(si, NE):
    when = when_q(si)
    # where = elf.where(si)
    why = why_q_main(si)
    who = who_main(si, NE)
    if when:
        print("	*** when or where : ", str(when))

    if why:
        print(" *** why  : ", str(why))
    if who:
        print(" *** www  : ", str(who))

### main Q/A generation function
def generate_q(corpus,NE):
    questions = []
    for si in corpus:
        when = when_q(si)
        why = why_q_main(si)
        who = who_main(si, NE)
        if when:
            # print("	*** when or where : ", str(when_where))
            questions.append(str(when))
        if why:
            # print(" *** why  : ", str(why))
            questions.append(str(why))
        if who:
            # print(" *** www  : ", str(what_who))
            questions.append(str(who))
    # clean wrong symbols in questions
    questions = [period_space_corr(question.replace('-LRB- ','(')).replace(' -RRB-',')')for question in questions]
    return questions