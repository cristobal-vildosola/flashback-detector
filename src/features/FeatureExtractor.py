from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, data):
        pass

    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def name(self) -> str:
        pass
