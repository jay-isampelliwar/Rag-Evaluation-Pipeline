class DocumentData:
    id: str
    content: str
    rank: int

    def __init__(self, id: str, content: str, rank: int) -> None:
        self.id = id
        self.content = content
        self.rank = rank

class SourceData:

    source : str
    page: int
    title: str
    authors: str

    def __init__(self, source: str, page: int, title: str, authors: str) -> None:
        self.source = source
        self.page = page
        self.title = title
        self.authors = authors


class RetrieverResponseModel:
    document_data: list[DocumentData]
    sources_data: list[SourceData]
    scores: list[float]

    def __init__(self, document_data: list[DocumentData], sources_data: list[SourceData], scores: list[float]):
        self.document_data = document_data
        self.sources_data = sources_data
        self.scores = scores
