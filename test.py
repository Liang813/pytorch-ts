import matplotlib.pyplot as plt
import pandas as pd
import torch

from pts.dataset import ListDataset
from pts.model.deepar import DeepAREstimator
from pts import Trainer
from pts.dataset import to_pandas


if __name__ == '__main__':
  url = "Twitter_volume_AMZN.csv"
  df = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
  
  training_data = ListDataset(
    [{"start": df.index[0], "target": df.value[:"2015-04-05 00:00:00"]}],
    freq = "5min"
  )
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  estimator = DeepAREstimator(freq="5min",
                              prediction_length=12,
                              input_size=43,
                              trainer=Trainer(epochs=10,
                                              device=device))
  predictor = estimator.train(training_data=training_data)
  

