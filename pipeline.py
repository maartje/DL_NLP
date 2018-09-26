import preprocess
import train
import predict
import evaluate
import tf_idf_baseline

def main():
    preprocess.main()
    train.main()
    predict.main()
    tf_idf_baseline.main()
    evaluate.main()

if __name__ == "__main__":
    main()
