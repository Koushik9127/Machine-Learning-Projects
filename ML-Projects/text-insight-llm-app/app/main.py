from src.preprocess import clean_text
from src.model import predict_insights
from src.utils import print_insights

def main():
    text = "This is a sample text to extract insights."
    cleaned = clean_text(text)
    insights = predict_insights(cleaned)
    print_insights(insights)

if __name__ == "__main__":
    main()
