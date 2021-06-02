from tqdm import tqdm
import vision_tools
import importlib

importlib.reload(vision_tools)

for i in tqdm(range(1,2)):
    folder = 'train_games/game' + str(i)
    for j in tqdm(range(1,2)):
        file = str(j) + '.jpg'
        vision_tools.card_pipeline(folder, file)
