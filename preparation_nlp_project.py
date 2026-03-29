import requests
import conllu
import pandas as pd
import re

# Reading in Welsh UD Treebank info - it's split into train, dev and test

url_train = "https://raw.githubusercontent.com/bethancunningham/tfm/main/treebank_train.conllu"
request_1 = requests.get(url_train)
train_file = request_1.text

url_dev = "https://raw.githubusercontent.com/bethancunningham/tfm/main/treebank_dev.conllu"
request_2 = requests.get(url_dev)
dev_file = request_2.text

url_test = "https://raw.githubusercontent.com/bethancunningham/tfm/main/treebank_test.conllu"
request_3 = requests.get(url_test)
test_file = request_3.text

# Merging 3 files because I don't need to train

files = [train_file, dev_file, test_file]

merged_tokenlists = []
for file in files:
    for tokenlist in conllu.parse(file):
        merged_tokenlists.append(tokenlist)

print("Total number of sentences initially:", len(merged_tokenlists))

# Creating dataframe with data from treebank. Searching for sentences with mutated words

data = []

for sentence in merged_tokenlists:
  sent_id = sentence.metadata.get("sent_id")
  text = sentence.metadata.get("text")

  for i, token in enumerate(sentence):
      feats = token.get("feats")

      if feats and "Mutation" in feats:
          correct_form = token["form"]
          lemma = token["lemma"]
          mutation_type = feats["Mutation"]
          sentence_en = sentence.metadata.get("text_en")
          prev_word = sentence[i-1]["form"] if i > 0 else None
          prev_lemma = sentence[i-1]["lemma"] if i > 0 else None
          prev_token = sentence[i-1] if i > 0 else None

          data.append({
                "sent_id": sent_id,
                "sentence": text,
                "sentence_tokens": [t["form"] for t in sentence],
                "sentence_en": sentence_en,
                "correct_word": token["form"],
                "lemma": token["lemma"],
                "mutation_type": feats["Mutation"],
                "token_id": token["id"],
                "prev_word": prev_word,
                "prev_lemma": prev_lemma,
                "prev_pos": prev_token["upos"] if prev_token else None,
                })

df = pd.DataFrame(data)

# Limiting to soft mutation and not h-prothesis (not mutation proper)

df = df[df["mutation_type"] == "SM"]
df = df[~df["correct_word"].str.startswith("h")] # not including h-prothesis
# print(df.head(20))


# Adding extra column with unmutated word

def unmutate(row):
    """Function to take word bearing soft mutation and change it back to its root form (unmutated)"""
    word = row["correct_word"].lower()
    mut = row["mutation_type"]
    lemma = row["lemma"].lower()

    if mut == "SM" :
        if lemma.startswith("g") or word == "orau" or word == "well" : return "g" + word # This is to get around the fact the initial g is removed in soft mutation. Can't search for letter that isn't there so searching for g at the start of the lemma. 'gwell' and 'gorau' are irregular cases
        elif lemma.startswith("m") :    return "m" + word[1:] # to differentiate between m to f and b to f
        elif word.startswith("f") :      return "b" + word[1:]
        elif word.startswith("dd") :   return "d" + word[2:]
        elif word.startswith("b") :    return "p" + word[1:]
        elif word.startswith("d") :    return "t" + word[1:]
        elif word.startswith("g") :    return "c" + word[1:]
        elif word.startswith("l") :    return "ll" + word[1:]
        elif word.startswith("r") :    return "rh" + word[1:]
        else : return None

df["incorrect_word"] = df.apply(unmutate, axis=1)

# Removing rows with no unmutated version as these are errors - not really cases of mutation 

df = df[df["incorrect_word"].notna()]

# Putting capital letter back if mutated word starts with capital letter

df["incorrect_word"] = df.apply(lambda row: row["incorrect_word"][0].upper() + row["incorrect_word"][1:] if row["correct_word"][0].isupper() else row["incorrect_word"], axis=1)

# print(df.head(20))

# Creating lists of lexical triggers and lexicalised mutations

lexical_triggers_SM = ["a", "ail", "am", "ambell", "amryw", "ar", "at", "ba", "cryn", "cwbl", "cyfryw", "cymharol", "cyn", "dacw", "dan", "dau", "ddau", "dros", "drwy", "dwy", "ddwy", "dy", "dyma", "dyna", "ei", "gan", "go", "gwbl", "gweddol", "heb", "holl", "hollol", "hyd", "hynod", "i", "lled", "mha", "mor", "na", "naill", "neu", "newydd", "ni", "o", "oll", "oni", "pa", "pan", "pedwar", "pedwaredd", "po", "pur", "pwy", "rhy", "rhyw", "saith", "sir", "tair", "tan", "tros", "trwy", "trydedd", "unrhyw", "weled", "wrth", "wyth", "ychydig", "ynteu", "'th", "'i", "'w", "'na", "'ma"]
lexicalised_mut = ["ddim", "ddoe", "bynnag", "beth", "fenni", "bontnewydd", "dan"]

def classify_trigger(row):
    """Function to identify soft mutation trigger (lexical or morphosyntactic)"""
    mut = row["mutation_type"]
    prev = row["prev_word"]
    prev_lemma = row["prev_lemma"]
    prev_pos = row["prev_pos"]
    form = row["correct_word"]

    if mut == "SM":
        if form.lower() in lexicalised_mut :
            return "LEXICALISED"
        elif prev and prev.lower() in lexical_triggers_SM:
            return "L"
        elif prev and prev.lower() in {"fe", "mi"} and prev_pos == "PART" : # fe and mi particles are triggers but not fe and mi pronouns
            return "L"
        elif prev and prev_lemma == "bod" : # checking for forms of bod
            return "L"
        else:
            return "MS"

    return None

df["trigger_type"] = df.apply(classify_trigger, axis=1)

# Removing lexicalised mutations from df

df = df[df["trigger_type"] != "LEXICALISED"]

print("Total no. of sentences after removal of h-prothesis and errors and lexicalised mutations", df["sent_id"].nunique())
print("No. instances of mutation after removal of h-prothesis and errors and lexicalised mutations:", len(df))
print(df.groupby(["mutation_type", "trigger_type"]).size())

# Creating sample of 100 rows MS trigger and 100 L trigger

sample_L = df[df["trigger_type"] == "L"].sample(n=100, random_state=123)
sample_MS = df[df["trigger_type"] == "MS"].sample(n=100, random_state=123)

# Bringing them together, frac=1 to shuffle the order

sample = pd.concat([sample_L, sample_MS]).sample(frac=1, random_state=123).reset_index(drop=True)

# Creating incorrect_sentence column with unmutated word replacing mutated word

sample["incorrect_sentence"] = sample.apply(
    lambda row: row["sentence"].replace(row["correct_word"], row["incorrect_word"]), axis=1
)

# print(sample.head(20))

# Creating df where the correct version of the word is unmutated, and the incorrect one is mutated
# I will use a sample of the same lexical items seen in my SM examples, but unmutated, by searching for sentences with these unmutated words in the whole Treebank 

unmutated_words = sample["incorrect_word"].tolist()
# print(unmutated_words)

unmutated_words_data = []

for sentence in merged_tokenlists:
  sent_id = sentence.metadata.get("sent_id")
  text = sentence.metadata.get("text")

  for i, token in enumerate(sentence):
      if token["form"] in unmutated_words:
          correct_form = token["form"]
          lemma = token["lemma"]
          mutation_type = feats["Mutation"] if feats else None
          sentence_en = sentence.metadata.get("text_en")
          prev_word = sentence[i-1]["form"] if i > 0 else None
          prev_lemma = sentence[i-1]["lemma"] if i > 0 else None
          prev_token = sentence[i-1] if i > 0 else None

          unmutated_words_data.append({
                "sent_id": sent_id,
                "sentence": text,
                "sentence_tokens": [t["form"] for t in sentence],
                "sentence_en": sentence_en,
                "correct_word": token["form"],
                "lemma": token["lemma"],
                "mutation_type": feats["Mutation"] if feats else None,
                "token_id": token["id"],
                "prev_word": prev_word,
                "prev_lemma": prev_lemma,
                "prev_pos": prev_token["upos"] if prev_token else None,
                })

df_no_mutation = pd.DataFrame(unmutated_words_data)
# print(len(df_no_mutation))
# print(df_no_mutation["correct_word"].head(20))

# Taking sample of 100 of the no_mutation examples 

sample_nomutation = df_no_mutation.sample(n=100, random_state=123)

# print(sample_nomutation.tail(20))
# print(len(sample_nomutation))

# Mutating unmutated words to create incorrect sentences

def mutate(row):
    """Function to take root (unmutated) word and return word bearing soft mutation"""
    word = row["correct_word"].lower()

    if word.startswith("g") :   return word[1:] # removing the g
    elif word.startswith("m") :    return "f" + word[1:]
    elif word.startswith("b") :      return "f" + word[1:]
    elif word.startswith("d") :   return "dd" + word[1:]
    elif word.startswith("p") :    return "b" + word[1:]
    elif word.startswith("t") :    return "d" + word[1:]
    elif word.startswith("c") :    return "g" + word[1:]
    elif word.startswith("ll") :    return "l" + word[2:]
    elif word.startswith("rh") :    return "r" + word[2:]
    else : return None

sample_nomutation["incorrect_word"] = sample_nomutation.apply(mutate, axis=1)
# print(sample_nomutation.head(10))

# Creating incorrect_sentence column with mutated word replacing unmutated word

sample_nomutation["incorrect_sentence"] = sample_nomutation.apply(
    lambda row: row["sentence"].replace(row["correct_word"], row["incorrect_word"]), axis=1
)

# print(sample_nomutation["incorrect_sentence"].head(10))

# Creating final df with MS samples, L samples and no_mutation samples

df_final = pd.concat([sample, sample_nomutation]).sample(frac=1, random_state=123).reset_index(drop=True)

df_final = df_final[["sent_id", "sentence", "sentence_en", "correct_word", "mutation_type", "trigger_type", "incorrect_word", "incorrect_sentence"]]

df_final["mutation_type"] = df_final["mutation_type"].fillna("no_mutation")
df_final["trigger_type"] = df_final["trigger_type"].fillna("no_mutation")

df_final.to_csv("data_nlp_3.csv", index=False)