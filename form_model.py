import sys
from modeling import train_and_evaluate

if __name__ == "__main__":
    train_and_evaluate("form", refresh="--refresh" in sys.argv)