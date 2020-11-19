import torch
import matplotlib.pyplot as plt
import numpy as np
from Utils.metrics import generate_metrics

from Modules.Projectors.michelot import michelot
from Modules.Projectors import duchi_sort
from Modules.Projectors import duchi_sort_numpy
from Modules.Projectors import condat
from Modules.Projectors import descent

# The following code can be used to test the validity of any of the following projectors
"""
from Utils.test_projector import test_projector
test_projector(descent.descent_l1) # Returns True if projections are valid, False if not
"""

results = generate_metrics(n=100, dim=25, projectors={
    "Duchi": duchi_sort.project_l1_ball_serial,
    "Michelot": michelot,
    "Condat": condat.condat_l1,
    # "Descent": descent.descent_l1
}, plot=True)

plt.show()