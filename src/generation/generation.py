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
        self.retrieval = None

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

        citations = [ f"{i + 1}. Title: {source.title} Document: {source.source} Authors: {source.authors}"
                      for i, source in enumerate(sources)]

        answer_with_citations = content + "\n\nCitations:\n" + "\n".join(citations) if citations else content

        return answer_with_citations