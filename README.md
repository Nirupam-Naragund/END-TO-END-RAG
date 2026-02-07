
<p align="center"><h1 align="center">Hybrid RAG with ColBERT, Sparse-Dense Retrieval, and Semantic Cache</h1></p>

<p align="center">
	<img src="https://img.shields.io/github/license/Nirupam-Naragund/END-TO-END-RAG?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/Nirupam-Naragund/END-TO-END-RAG?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Nirupam-Naragund/END-TO-END-RAG?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/Nirupam-Naragund/END-TO-END-RAG?style=default&color=0080ff" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>



---

##  Overview


A high-performance RAG pipeline that leverages Llama 3.1 (via Groq), Qdrant for multi-vector hybrid search, and Redis for semantic caching. This architecture is designed to handle complex queries by combining the semantic depth of dense embeddings, the keyword precision of sparse vectors (SPLADE), and the token-level nuance of ColBERT.


---

##  Features


** Triple-Hybrid Retrieval:

- ** Dense: BGE-Small-en-v1.5 for broad semantic meaning.

- ** Sparse: Splade PP for exact keyword matching.

- ** ColBERT: Late interaction multi-vector retrieval for high-precision ranking.

- ** Semantic Caching: Powered by RedisVL, reducing LLM costs and latency by caching semantically similar queries.

- ** Fast Inference: Integrated with Groq for ultra-fast Llama 3.1 generation.

- ** Multi-format Support: Automated ingestion for .pdf, .docx, .txt.


---

##  Project Structure

```sh
└── END-TO-END-RAG/
    ├── data
    │   └── Attention.pdf
    ├── main.py
    └── requirements.txt
```



---
##  Getting Started

###  Prerequisites

- ** Python: 3.9+

- ** Qdrant: Running locally at http://localhost:6333 (or Docker).

- ** Redis: Running locally at redis://localhost:6379.

- ** API Keys: A Groq Cloud API Key.


###  Installation

Install END-TO-END-RAG using one of the following methods:

**Build from source:**

1. Clone the END-TO-END-RAG repository:
```sh
❯ git clone https://github.com/Nirupam-Naragund/END-TO-END-RAG
```

2. Navigate to the project directory:
```sh
❯ cd END-TO-END-RAG
```

3. Install the project dependencies:


**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r requirements.txt
```




###  Usage
Run END-TO-END-RAG using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ python main.py
```


##  Contributing

- **💬 [Join the Discussions](https://github.com/Nirupam-Naragund/END-TO-END-RAG/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/Nirupam-Naragund/END-TO-END-RAG/issues)**: Submit bugs found or log feature requests for the `END-TO-END-RAG` project.
- **💡 [Submit Pull Requests](https://github.com/Nirupam-Naragund/END-TO-END-RAG/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/Nirupam-Naragund/END-TO-END-RAG
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/Nirupam-Naragund/END-TO-END-RAG/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=Nirupam-Naragund/END-TO-END-RAG">
   </a>
</p>
</details>

---

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---
