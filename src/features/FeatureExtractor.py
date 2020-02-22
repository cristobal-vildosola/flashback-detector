from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, data):
        pass
