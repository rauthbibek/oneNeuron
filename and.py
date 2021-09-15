from utils.models import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
from utils.constants import AND, EPOCHS, ETA
import pandas as pd
import numpy as np

def main(AND, EPOCHS, ETA):

    df = pd.DataFrame(AND)
    print(df)
    X,y = prepare_data(df)
    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)
    _ = model.total_loss()
    save_model(model, filename="and.model")
    save_plot(df,"and.png", model)

if __name__=='__main__': ##entry point
    main(AND,EPOCHS,ETA)