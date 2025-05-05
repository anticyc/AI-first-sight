from src.dataProcessor import DataProcessor
from src.bigram import main
# run Data Processor
if __name__ == '__main__':
    dp = DataProcessor("corpus/sina_news_gbk")
    dp.do(False)
    dp.output("data/temp", False)
    main()
    