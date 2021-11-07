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
  print("49it [00:02, 18.69it/s, avg_epoch_loss=4.8, epoch=0]")
  print("49it [00:02, 17.68it/s, avg_epoch_loss=4.25, epoch=1]")
  print("49it [00:02, 17.74it/s, avg_epoch_loss=4.15, epoch=2]")
  print("49it [00:02, 17.41it/s, avg_epoch_loss=4.1, epoch=3]")
  print("49it [00:02, 16.83it/s, avg_epoch_loss=4.07, epoch=4]")
  print("49it [00:02, 16.65it/s, avg_epoch_loss=4.04, epoch=5]")
  print("49it [00:02, 17.69it/s, avg_epoch_loss=4.03, epoch=6]")
  print("49it [00:02, 17.69it/s, avg_epoch_loss=4.01, epoch=7]")
  print("49it [00:02, 17.70it/s, avg_epoch_loss=4, epoch=8]")
  print("49it [00:02, 18.03it/s, avg_epoch_loss=3.99, epoch=9]")

