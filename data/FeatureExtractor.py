

class FeatureExtractor(object):
    """
    Parent class for a general feature extractor (passed to data loader).
    """
    def __init__(self):
        pass

    def extract_feature(self, *args, **kwargs):
        """
        Must return PyTorch Variable object.
        :param args:
        :param kwargs:
        :return:
        """
        pass
