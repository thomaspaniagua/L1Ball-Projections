import torch
import matplotlib.pyplot as plt
import numpy as np

reimporter.Reimport()

from Utils.metrics import generate_metrics

results = generate_metrics(n=100, dim=25, projectors={
    "Duchi": duchi_sort.project_l1_ball_serial,
    "Michelot": michelot,
    "Condat": condat.condat_l1,
    # "Descent": descent.descent_l1
}, plot=True)