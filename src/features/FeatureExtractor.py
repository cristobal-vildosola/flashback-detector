from abc import ABC, abstractmethod
import numpy


class FeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, data) -> numpy.ndarray:
        pass

    @abstractmethod
    def descriptor_size(self) -> int:
        pass

    @abstractmethod
    def name(self) -> str:
        pass
