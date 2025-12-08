
import pathlib
import  numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM , Dense , Activation, Input
from tensorflow.keras.optimizers import RMSprop

filepath = pathlib.Path("Prompt_Generator/shakespeare.txt")
text = open(filepath,"rb").read().decode("utf-8").lower()

character = sorted(set(text))

char_to_index = dict((c,i) for i , c in enumerate(character))
index_to_char = dict((i,c) for i , c in enumerate(character))


seq_length = 40
step_size = 3

sentences = []   # --> feature
next_char = []   # --> target


for i in range(0,len(text) - seq_length,step_size):
    sentences.append(text[i : i + seq_length])
    next_char.append(text[i+seq_length])


x = np.zeros((len(sentences),seq_length,len(character)),dtype=bool)
y = np.zeros((len(sentences),len(character)),dtype=bool)


for i , sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i,t,char_to_index[char]] = 1
    y[i,char_to_index[next_char[i]]] = 1


model = Sequential()
model.add(Input(shape=(seq_length,len(character))))
model.add(LSTM(128))
model.add(Dense(len(character)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate= 0.01))
model.fit(x,y,batch_size = 256,epochs=5)

model.save("textgen.keras")


model = tf.keras.models.load_model('Prompt_Generator/textgen.keras')

## helper function
def sample(preds,temperature = 0.6):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)/temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1,preds,1)
    return np.argmax(probas)

def generateText(seed_text,length,temperature):
    sentence = seed_text.lower()
    if len(sentence) < seq_length:
        padding = ' ' * (seq_length - len(sentence))
        sentence = padding + sentence
    elif len(sentence) > seq_length:
        sentence = sentence[-seq_length:]

    generated = sentence

    for i in range(length):
        x = np.zeros((1,seq_length,len(character)))
        for t , char in enumerate(sentence):
            x[0,t,char_to_index[char]] = 1

        prediction = model.predict(x,verbose=0)[0]
        next_index = sample(prediction,temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated


def make_centered_banner(title: str) -> str:
    padding_left_right = 2
    inner_width = len(title) + padding_left_right * 2
    top_border    = "+" + "=" * (inner_width + 4) + "+"
    empty_line    = "||" + " " * (inner_width + 0) + "||"
    title_line = "||" + title.center(inner_width) + "||"
    bottom_border = "+" + "=" * (inner_width + 4) + "+"
    return "\n".join([top_border, empty_line, title_line, empty_line, bottom_border])


if __name__ == "__main__":
    print(make_centered_banner("Prompt Generator"))
    print("")

my_prompt = input("enter your prompt >> \n")
generated_text = generateText(
    seed_text=my_prompt,
    length=300,
    temperature=0.75
)
print("==================  OUTPUT PROMPT========================")
print(generated_text)