import os
import shutil
import json
from pathlib import Path
from langchain_community.document_loaders import (
    UnstructuredExcelLoader,
    UnstructuredCSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Union, Optional
import numpy as np
from app.common.Image2Text import ImageParser
from app.common.config import Config
import shutil


class VectorDB:
    def __init__(self, embeddings: OpenAIEmbeddings, index_path: str="./db/flowbox", keyword_file: str="./keywords.json"):
        self.index_path = index_path
        self.vector_store: Optional[FAISS] = None
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.keyword_file = keyword_file
        self.keywords = self._load_keywords()

    def _load_keywords(self):
        """加载本地关键字文件"""
        if os.path.exists(self.keyword_file):
            with open(self.keyword_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_keywords(self):
        """保存关键字到本地文件"""
        with open(self.keyword_file, 'w', encoding='utf-8') as f:
            json.dump(self.keywords, f, ensure_ascii=False, indent=4)

    def _extract_keywords(self, content: str, source: str) -> List[str]:
        """使用 LLM 提取关键字"""
        llm = ChatOpenAI(base_url=Config.CHATGPT_BASE_URL, model=Config.CHATGPT_MODEL, api_key=Config.CHATGPT_API_KEY)
        prompt = PromptTemplate(
            input_variables=["content"],
            template="从以下文本中提取关键词，返回列表格式：\n{content}"
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        try:
            response = llm_chain.run(content=content[:1000])  # 限制长度
            keywords = eval(response) if response.startswith('[') else response.split()
            self.keywords[source] = keywords
            self._save_keywords()
            print(f"提取关键字成功: {source} -> {keywords}")
            return keywords
        except Exception as e:
            print(f"提取关键字失败: {str(e)}")
            return []

    def select(self):
        try:
            index_path = Path(self.index_path)
            if not index_path.exists():
                raise FileNotFoundError(f"FAISS 索引路径 {self.index_path}不存在")
            self.vector_store = FAISS.load_local(
                folder_path=self.index_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"成功加载 FAISS 索引,当前向量数量: {self.vector_store.index.ntotal}")
        except Exception as e:
            print(f"加载FAISS索引失败: {str(e)}")
            raise

    def create(self, documents: List[Union[Document, str]]) -> None:
        try:
            if not documents:
                raise ValueError("文档列表为空，无法创建FAISS索引")
            docs = []
            for item in documents:
                if isinstance(item, str):  # 如果是文件路径，加载为 Document
                    file_path = str(Path(item))
                    suffix = Path(file_path).suffix.lower()
                    loader_map = {
                        ".xlsx": UnstructuredExcelLoader,
                        ".xls": UnstructuredExcelLoader,
                        ".csv": UnstructuredCSVLoader,
                        ".md": UnstructuredMarkdownLoader,
                        ".pdf": PyPDFLoader,
                        ".doc": UnstructuredWordDocumentLoader,
                        ".docx": UnstructuredWordDocumentLoader,
                        ".txt": TextLoader,
                        ".ppt": UnstructuredPowerPointLoader,
                        ".pptx": UnstructuredPowerPointLoader,
                        ".json": TextLoader,
                        ".xml": TextLoader,
                        ".yaml": TextLoader
                    }
                    loader_class = loader_map.get(suffix)
                    if not loader_class:
                        raise ValueError(f"不支持的文件格式: {suffix}")
                    loader = loader_class(file_path)
                    loaded_docs = loader.load()
                    if not loaded_docs:
                        raise ValueError(f"未加载到 {file_path} 的内容")
                    for doc in loaded_docs:
                        if "source" not in doc.metadata or not doc.metadata["source"]:
                            doc.metadata["source"] = file_path
                        keywords = self._extract_keywords(doc.page_content, doc.metadata["source"])
                        doc.metadata["keywords"] = keywords
                        docs.extend(self.text_splitter.split_documents([doc]))
                elif isinstance(item, Document):  # 如果已经是 Document 对象
                    doc = item
                    if "source" not in doc.metadata or not doc.metadata["source"]:
                        doc.metadata["source"] = "未知来源"
                    keywords = self._extract_keywords(doc.page_content, doc.metadata["source"])
                    doc.metadata["keywords"] = keywords
                    docs.extend(self.text_splitter.split_documents([doc]))
                else:
                    raise ValueError(f"不支持的输入类型: {type(item)}")
            texts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            ids = [str(i) for i in range(len(docs))]
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
                ids=ids
            )
            self.vector_store.save_local(self.index_path)
            print(f"成功创建FAISS索引，存储到{self.index_path},向量数量：{self.vector_store.index.ntotal}")
        except Exception as e:
            print(f"创建 FAISS 索引失败: {str(e)}")
            raise

    def query(self, query_text: str, k: int=5) -> List[Document]:
        try:
            if self.vector_store is None:
                raise ValueError("FAISS 索引未初始化，(select/create后才能执行query操作)")
            results = self.vector_store.similarity_search(query_text, k=k)
            print(f"查询返回 {len(results)}个结果")
            return results
        except Exception as e:
            print(f"查询 FAISS 索引失败: {str(e)}")
            raise

    def delete(self, ids: List[str]) -> None:
        try:
            if self.vector_store is None:
                raise ValueError("FAISS 索引未初始化，请先调用select、create")
            if not ids:
                raise ValueError("ID 列表未空，无法删除")
            ids_array = np.array([int(id_) for id_ in ids], dtype=np.int64)
            self.vector_store.index.remove_ids(ids_array)
            self.vector_store.save_local(self.index_path)
            print(f"成功删除{len(ids)}个向量， 当前向量数量：{self.vector_store.index.ntotal}")
        except Exception as e:
            print(f"删除 FAISS 向量失败: {str(e)}")
            raise

    def insert(self, file_path: str) -> None:
        try:
            if self.vector_store is None:
                raise ValueError("FAISS 索引未初始化，请先调用select/create")
            print(f"加载新文档: {file_path}")
            file_path = str(Path(file_path))
            suffix = Path(file_path).suffix.lower()
            loader_map = {
                ".xlsx": UnstructuredExcelLoader,
                ".xls": UnstructuredExcelLoader,
                ".csv": UnstructuredCSVLoader,
                ".md": UnstructuredMarkdownLoader,
                ".pdf": PyPDFLoader,
                ".doc": UnstructuredWordDocumentLoader,
                ".docx": UnstructuredWordDocumentLoader,
                ".txt": TextLoader,
                ".ppt": UnstructuredPowerPointLoader,
                ".pptx": UnstructuredPowerPointLoader,
                ".json": TextLoader,
                ".xml": TextLoader,
                ".yaml": TextLoader
            }
            loader_class = loader_map.get(suffix)
            if not loader_class:
                raise ValueError(f"不支持的文件格式: {suffix}")
            loader = loader_class(file_path)
            documents = loader.load()
            if not documents:
                raise ValueError("未加载到新文档内容")

            for doc in documents:
                if "source" not in doc.metadata or not doc.metadata["source"]:
                    doc.metadata["source"] = file_path
                keywords = self._extract_keywords(doc.page_content, doc.metadata["source"])
                doc.metadata["keywords"] = keywords

            new_docs = self.text_splitter.split_documents(documents)
            print(f"分割后新文档块数量: {len(new_docs)}")

            new_texts = [doc.page_content for doc in new_docs]
            new_metadatas = [doc.metadata for doc in new_docs]
            print(f"提取文本数量: {len(new_texts)}")

            current_num_vectors = self.vector_store.index.ntotal
            new_ids = [str(current_num_vectors + i) for i in range(len(new_docs))]
            self.vector_store.add_texts(
                texts=new_texts,
                metadatas=new_metadatas,
                ids=new_ids
            )
            print(f"已添加 {len(new_docs)} 个新文档到 FAISS 索引")

            self.vector_store.save_local(self.index_path)
            print(f"更新后的 FAISS 索引已保存到 {self.index_path}")
        except Exception as e:
            print(f"插入新文档到 FAISS 失败: {str(e)}")
            raise

    def insert_image(self, image_path: str) -> None:
        """插入图像并提取描述"""
        try:
            if self.vector_store is None:
                raise ValueError("FAISS 索引未初始化，请先调用select/create")
            print(f"加载新图像: {image_path}")
            image_path = str(Path(image_path))
            if not os.path.isfile(image_path):
                raise ValueError(f"图像文件 {image_path} 不存在")

            # 使用 ImageParser 提取描述
            img_parser = ImageParser()
            description = img_parser.extract(image_path)
            if not description:
                raise ValueError("图像描述提取失败")

            # 创建 Document 对象
            doc = Document(
                page_content=description,
                metadata={"source": image_path, "type": "image"}
            )
            keywords = self._extract_keywords(description, image_path)
            doc.metadata["keywords"] = keywords

            new_docs = self.text_splitter.split_documents([doc])
            print(f"分割后新图像描述块数量: {len(new_docs)}")

            new_texts = [doc.page_content for doc in new_docs]
            new_metadatas = [doc.metadata for doc in new_docs]
            current_num_vectors = self.vector_store.index.ntotal
            new_ids = [str(current_num_vectors + i) for i in range(len(new_docs))]
            self.vector_store.add_texts(
                texts=new_texts,
                metadatas=new_metadatas,
                ids=new_ids
            )
            print(f"已添加 {len(new_docs)} 个图像描述到 FAISS 索引")

            self.vector_store.save_local(self.index_path)
            print(f"更新后的 FAISS 索引已保存到 {self.index_path}")
        except Exception as e:
            print(f"插入图像到 FAISS 失败: {str(e)}")
            raise

    def set_embeddings(self, embeddings: OpenAIEmbeddings) -> None:
        if self.vector_store is not None:
            print("警告: 更改embeddings 可能导致索引不一致，请重新调用select 或 create")
        self.embeddings = embeddings

# 支持的文件格式
SUPPORTED_EXTENSIONS = [
    "*.txt", "*.csv", "*.md",
    "*.pdf", "*.doc", "*.docx", "*.ppt", "*.pptx",
    "*.xlsx", "*.xls",
    "*.json", "*.xml", "*.yaml"
]

def traverse_files(path):
    """遍历目录，返回支持的文件列表"""
    files = []
    path_obj = Path(path)
    if not path_obj.is_dir():
        raise ValueError(f"路径 {path} 不是有效目录")

    for ext in SUPPORTED_EXTENSIONS:
        for file_path in path_obj.rglob(ext):
            files.append(str(file_path))
    return files


def cutAndPaste(source_path, destination_dir):
    try:
        source_path = str(Path(source_path))
        destination_dir = str(Path(destination_dir))
        os.makedirs(destination_dir, exist_ok=True)
        destination_path = os.path.join(destination_dir, os.path.basename(source_path))
        shutil.move(source_path, destination_path)
        return destination_path
    except Exception as e:
        raise Exception(f"文件移动失败: {e}")
