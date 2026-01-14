
# import
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,models,Input
from matplotlib import pyplot as plt
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# load dataset
ds = load_dataset("sgtsaughter/pokemon-classification-images-151")

dataset = ds['train']
items = dataset[0]
training_images = dataset['image']
training_labels = dataset['label']

# converting to np.array

# final_training_image = []
# for image in training_images:
#     img = image.resize((224,224))
#     img = np.array(img)/255.0
#     final_training_image.append(img)
IMG_SIZE = 32
final_training_image = []
for img in tqdm(dataset['image'], desc="Converting images"):
    if not isinstance(img, Image.Image):
        # dataset may store bytes or dicts; handle common case where img is PIL already
        img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    img = img.convert("RGB")            # ensure 3 channels
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    final_training_image.append(arr)

X = np.stack(final_training_image)     # shape: (N, 224,224,3)
y = np.array(dataset['label']).astype(np.int32)


class_name = [
    # 000-019
    "Abra", "Aerodactyl", "Alakazam", "Arbok", "Arcanine", "Articuno",
    "Beedrill", "Bellsprout", "Blastoise", "Bulbasaur", "Butterfree", "Caterpie",
    "Chansey", "Charizard", "Charmander", "Charmeleon", "Clefable", "Clefairy",
    "Cloyster", "Cubone",

    # 020-039
    "Dewgong", "Diglett", "Ditto", "Dodrio", "Doduo", "Dragonair",
    "Dragonite", "Dratini", "Drowzee", "Dugtrio", "Eevee", "Ekans",
    "Electabuzz", "Electrode", "Exeggcute", "Exeggutor", "Farfetch'd", "Fearow",
    "Flareon", "Gastly",

    # 040-059
    "Gengar", "Geodude", "Gloom", "Golbat", "Goldeen", "Golduck",
    "Golem", "Graveler", "Grimer", "Growlithe", "Gyarados", "Haunter",
    "Hitmonchan", "Hitmonlee", "Horsea", "Hypno", "Ivysaur", "Jigglypuff",
    "Jolteon", "Jynx",

    # 060-079
    "Kabuto", "Kabutops", "Kadabra", "Kakuna", "Kangaskhan", "Kingler",
    "Koffing", "Krabby", "Lapras", "Lickitung", "Machamp", "Machoke",
    "Machop", "Magikarp", "Magmar", "Magnemite", "Magneton", "Mankey",
    "Marowak", "Meowth",

    # 080-099
    "Metapod", "Mew", "Mewtwo", "Mr. Mime", "Moltres", "Muk",
    "Nidoking", "Nidoqueen", "Nidoran♀", "Nidoran♂", "Nidorina", "Nidorino",
    "Ninetales", "Oddish", "Omanyte", "Omastar", "Onix", "Paras",
    "Parasect", "Persian",

    # 100-119
    "Pidgeot", "Pidgeotto", "Pidgey", "Pikachu", "Pinsir", "Poliwag",
    "Poliwhirl", "Poliwrath", "Ponyta", "Porygon", "Primeape", "Psyduck",
    "Raichu", "Rapidash", "Raticate", "Rattata", "Rhydon", "Rhyhorn",
    "Seadra", "Seaking",

    # 120-139
    "Seel", "Shellder", "Slowbro", "Slowpoke", "Snorlax", "Spearow",
    "Squirtle", "Starmie", "Staryu", "Scyther", "Tangela", "Tauros",
    "Tentacool", "Tentacruel", "Vaporeon", "Venusaur", "Venomoth", "Venonat",
    "Victreebel", "Vileplume",

    # 140-150
    "Voltorb", "Vulpix", "Wartortle", "Weedle", "Weepinbell", "Weezing",
    "Wigglytuff", "Zapdos", "Zubat"
]

# building model
model = models.Sequential()
model.add(Input(shape=(IMG_SIZE,IMG_SIZE,3)))
model.add(layers.Conv2D(32,(3,3),activation='relu',))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(151,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X,y,epochs=5,batch_size=4,validation_split=0.2)



