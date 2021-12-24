# multilabel_classification

根據景點評論，將景點分類為「景點」／「餐廳」／「住宿」（可多選） \

Fine-tuned with Bert\
No preprocess \
Concat 5 reviews of each spot from Google Map to form a training data

# Setup

Switch to GPU if needed, current setup using CPU

1. Run ```pip install -r requirements.txt``` in terminal
2. Run ```python main.py``` in terminal
3. Model will be in ```saved_model/chi_model.pt```
4. Confusion matrix analysis will be in ```saved_model/chi_model.pt```
5. Load model with ```load_model.py``` (currently executable with example file)
6. Observe dataset with ```observe_dataset.py```, result will be in ```data/frequency_distribution.png```
