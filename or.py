from utils.models import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
from utils.constants import OR, EPOCHS, ETA
import pandas as pd
import numpy as np
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logging_dir = "logs"
os.makedirs(logging_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(logging_dir,"or.log"),format=logging_str, level=logging.INFO, filemode="a")

def main(OR, EPOCHS, ETA):

    df = pd.DataFrame(OR)
    logging.info(f"This is actual dataframe {df}")
    X,y = prepare_data(df)
    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)
    _ = model.total_loss()
    save_model(model, filename="or.model")
    save_plot(df,"or.png", model)

if __name__=='__main__': ##entry point
    try:
        logging.info(">>>>>>> Starting training >>>>>>>>>>")
        main(OR,EPOCHS,ETA,)
        logging.info("<<<<<<< Training finished <<<<<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e