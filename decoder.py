# Seq2Seq decoder model

import torch.nn as nn
import torch.nn.functional as F

class AutoCompleteDecoderModel(nn.Module):

