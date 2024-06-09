
# VTB-align

Hey! Welcome to our fun projects of trying to aligning models with VTB values. I believe that this world is now deranged because people either don't know, or don't want to follow the cute rules of our kawaii vtbs (Chinese and Japanese vtb specifically). And this model will serve as a starting point to help make the world a better place by enforcing the VTB values to the world 💗




## What’s This All About?

so in this project we use a crawler and download some funny (but also trolling) comments from bilibili (specifically comments that are 整活，而且还得是好活), and use them as our SFT dataset. We use this dataset to perform SFT on our base model Qwen2-0.5b (as a start, we are also tring to play with Qwen2 of different size). We are currently trying to create our own preference data, and through that we will hopefully go through the whole aligning process to come up with the final model. 

### but seriously, these comments are only for training purpose, and we only use the comments along with timestamps and likes to formulate our dataset, and we use it for only educational and entertaining purpose. hopefully this doesnt violate any laws, if it does, so be it 😭


## How do I play with it?

The first temp model will be pushed to huggingface hub (after i finish writing this awesome README) but im not going to share it here. I should be able to come up with a html page where we can interact with the model soon, and it's link should be here: [Link](http://lindseyfeng.github.io) (for now, this link only takes you to my hand crafted awesome personal website that i didn't update for like a year now)


## Sample output

If you are intrigues already, good for you! Here are some fun generation (from a very preliminary model) for you to enjoy: 

![screenshot-1](./images/1.jpg)

## If you want to read my crappy code

`finetuning.py` contains a basic SFT code that you can find anywhere on the internet, while `inference.py` plays with the model trained by `fintuning.py`. `train_set.csv` and `test_set.csv` contains a part of the training set and testing set that we are using right now. 