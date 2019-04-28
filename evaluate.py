import sys

from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config



def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build("train")
    model.restore_session(config.dir_model)

    # create dataset
    if len(sys.argv) == 2:
        if sys.argv[1] == 'test':
            test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, max_length=None)
        elif sys.argv[1] == 'dev':
            test = CoNLLDataset(config.filename_dev, config.processing_word,
                                config.processing_tag, max_length=None)
    else:
        assert len(sys.argv) == 1
        test = CoNLLDataset(config.filename_test, config.processing_word,
                            config.processing_tag, max_length=None)
    # evaluate and interact
    model.evaluate(test)
    # interactive_shell(model)


if __name__ == "__main__":
    main()
