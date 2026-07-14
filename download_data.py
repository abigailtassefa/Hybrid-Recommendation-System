from dotenv import load_dotenv
import os


load_dotenv()
print(os.getenv('KAGGLE_USERNAME'))  # sanity check — should print abigailassefa
print(os.getenv('KAGGLE_KEY'))       # should print your key

#the dataset is publicly available on kaggle, so we can use the kagglehub package to download it. and it doesnt require the user to have a kaggle account or API key. it will download the dataset to a temporary directory and return the path to the downloaded files.
import kagglehub
path = kagglehub.dataset_download("jessicali9530/kuc-hackathon-winter-2018")
print("Path to dataset files:", path)