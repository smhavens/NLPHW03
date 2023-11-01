from datasets import load_dataset
dataset = load_dataset("wikipedia", "20220301. simple")
# check the first example of the training portion of the dataset :
print(dataset['train'][0])