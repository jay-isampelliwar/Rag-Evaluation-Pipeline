from langchain_openai import ChatOpenAI
from model.retriever_response_model import RetrieverResponseModel, SourceData
import yaml


class Generation:

    def __init__(self, model_name: str = "qwen/qwen3-vl-4b"):
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url="http://127.0.0.1:1234/v1/",
            api_key="",
        )

    def invoke(self, user_query: str, result_response: RetrieverResponseModel):
        try:

            prompt_config = self._load_prompt_config()

            prompt_template = prompt_config["prompt_template"]
            context = "\n\n".join(item.content for item in result_response.document_data)

            prompt = prompt_template.format(
                context=context,
                question=user_query,
            )

            llm_response = self.llm.invoke(prompt)
            answer_with_citations = self._add_citations(content=llm_response.content, sources=result_response.sources_data)

            return answer_with_citations

        except Exception as e:
            print("An error occured while invoking Agent")
            raise

    def _load_prompt_config(self, path: str = "configs/prompt.yaml"):
        with open(path, "r") as f:
            return yaml.safe_load(f)



    def _add_citations(self, content: str, sources: list[SourceData]) -> str:
        def _safe(s) -> str:
            if s is None or (isinstance(s, str) and not s.strip()):
                return "Unknown"
            return str(s).strip()

        seen = set()
        unique_sources = []
        for source in sources:
            title = _safe(source.title)
            doc = _safe(source.source)
            authors = _safe(source.authors)
            page = source.page
            if page is None or (isinstance(page, str) and not str(page).strip()):
                page_str = "Unknown"
            else:
                page_str = str(page)
            key = (title, doc)
            if key in seen:
                continue
            seen.add(key)
            unique_sources.append((title, doc, authors, page_str))

        if not unique_sources:
            return content

        lines = []
        for i, (title, doc, authors, page_str) in enumerate(unique_sources, 1):
            lines.append(f"{i}. Title: {title} | Document: {doc} | Page: {page_str} | Authors: {authors}")

        return content + "\n\n**Citations:**\n" + "\n".join(lines)