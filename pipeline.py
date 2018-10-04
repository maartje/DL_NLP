import numpy as np
import torch
import random

import preprocess
import train
import predict
import evaluate
import tf_idf_baseline

random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)

def main():
    print('*****************\n* preprocessing *\n*****************')
    preprocess.main()
    print('*****************\n*   training    *\n*****************')
    train.main()    
    print('*****************\n*  predicting   *\n*****************')
    predict.main()
    print('*****************\n*    ntf_idf    *\n*****************')
    tf_idf_baseline.main()
    print('*****************\n*  evaluation   *\n*****************')
    evaluate.main()

if __name__ == "__main__":
    main()