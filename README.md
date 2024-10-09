# A Graph Neural Network for Facial Palsy and Paresis Evaluation

This is the repo dedicated to PRHA 2024 workshop paper "A Graph Neural Network for Facial Palsy and Paresis Evaluation", held at ICPR 2024.

### Environment

#### Python version: 3.8

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset directory

Dataset is prepared so that there are directories for each class, with the image frames named following "{patient_id}_{frame_id}" structure.

```
...
|
|---nonstroke
|       |--subject0_frame0.png
|       |--subject0_frame0.png
|       | ...
|       |--subject1_frame0.png
|       |...
|
|---stroke
        |--patient0_frame0.png
        |--patient0_frame0.png
        | ...
        |--patient1_frame0.png
        |...
```

## Facial Graph Generation

Run ```graphizer478.py``` after editing the dataset directory to generate and save facial graphs.

## Training

Run ```main.py``` to train the model. Change training dataset directory as needed. For experiments with Toronto NeuroFace dataset, we use:
* 5 data splits for training on the whole dataset (option ```--type 'mixed'```)
* 4 data splits for training on "clean pose" dataset (option ```--type 'clean'```)
* 3 data splits for training on bad pose dataset (option ```--type 'pose'```)
\
\
The corresponding commands to train the model:
* ```bash
python main.py --type 'mixed' --epoch {NUMBER_EPOCH} --lr {LEARNING_RATE} --save_model {SAVED_WEIGHT_NAME}```
* ```bash
python main.py --type 'clean' --epoch {NUMBER_EPOCH} --lr {LEARNING_RATE} --save_model {SAVED_WEIGHT_NAME}```
* ```bash
python main.py --type 'pose' --epoch {NUMBER_EPOCH} --lr {LEARNING_RATE} --save_model {SAVED_WEIGHT_NAME}```

#### Evaluation

```bash
python validate.py --load_model {SAVED_WEIGHT_NAME}```
