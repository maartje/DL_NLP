import preprocess
import train
import predict
import evaluate

def main():
    preprocess.main()
    train.main()
    predict.main()
    evaluate.main()

if __name__ == "__main__":
    main()
