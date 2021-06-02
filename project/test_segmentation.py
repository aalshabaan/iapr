from tqdm import tqdm
import vision_tools
import importlib

# importlib.reload(vision_tools)

for i in range(1, 8):
    folder = 'train_games/game' + str(i)
    for j in range(1, 14):
        file = str(j) + '.jpg'
        vision_tools.card_pipeline(folder, file, verbose=True)

# vision_tools.card_pipeline('train_games/game5', '9.jpg', verbose=True)
