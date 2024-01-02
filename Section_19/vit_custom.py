#%% source
## https://medium.com/@yanis.labrak/how-to-train-a-custom-vision-transformer-vit-image-classifier-to-help-endoscopists-in-under-5-min-2e7e4110a353
import pandas
# %% packages 
from hugsvision.dataio.VisionDataset import VisionDataset 
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer 
from transformers import ViTFeatureExtractor, ViTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference

import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
# %% data prep 
train, val, id2label, label2id = VisionDataset.fromImageFolder(
    "./train/",
    test_ratio = 0.15,
    balanced = True, 
    augmentation = True, 
    torch_vision = False
)

# %%
huggingface_model = 'google/vit-base-patch16-224-in21k'

trainer = VisionClassifierTrainer(
    model_name = "MyDogClassifier",
    train = train,
    test = val,
    output_dir = './out/',
    max_epochs = 20,
    batch_size = 4,
    lr = 2e-5,
    fp16 = False,
    model = ViTForImageClassification.from_pretrained(
        huggingface_model,
        num_labels = len(label2id),
        label2id = label2id,
        id2label = id2label
    ),
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        huggingface_model,
    ),
)

# %%
