from abc import ABC, abstractmethod


class SearchIndex(ABC):

    @abstractmethod
    def neighbours_retrieved(self) -> int:
        pass

    @abstractmethod
    def candidate_count(self, vector):
        pass

    @abstractmethod
    def search(self, vector):
        pass

    @abstractmethod
    def name(self) -> str:
        pass
