# Knowledge Graph Question Answering System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

A sophisticated question-answering system that leverages knowledge graphs and natural language processing to provide accurate and contextual answers to user queries. This system combines the power of graph databases with advanced NLP techniques to deliver intelligent responses to complex questions.

## 🌟 Features

- **Entity Recognition**: Advanced entity extraction from questions using SpaCy
- **Knowledge Graph Integration**: Seamless integration with Neo4j for efficient information retrieval
- **Question Classification**: Intelligent categorization of questions into factual, explanatory, temporal/spatial, and general types
- **Semantic Similarity**: BERT-based semantic matching for improved answer relevance
- **RESTful API**: FastAPI-powered API for easy integration
- **Confidence Scoring**: Transparent confidence metrics for each answer
- **Extensible Architecture**: Modular design for easy extension and customization

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Neo4j Database
- Spacy English language model

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kg-qa-system.git
cd kg-qa-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Running the System

1. Start Neo4j database server

2. Launch the API server:
```bash
python src/api.py
```

3. Access the API at `http://localhost:8000`

## 📚 API Documentation

### Endpoints

- `POST /ask`: Submit a question
  ```bash
  curl -X POST "http://localhost:8000/ask" \
       -H "Content-Type: application/json" \
       -d '{"text": "What is the relationship between entity A and entity B?"}'
  ```

- `GET /health`: Check API health status

Interactive API documentation is available at `http://localhost:8000/docs`

## 🏗️ Project Structure

```
kg-qa-system/
├── config/
│   └── config.py          # Configuration management
├── data/                  # Data storage
├── src/
│   ├── api.py            # FastAPI application
│   ├── qa_system.py      # Core QA system
│   ├── database/
│   │   └── neo4j_client.py  # Neo4j database client
│   └── nlp/
│       └── processor.py  # NLP processing utilities
├── tests/                # Test suite
├── .env                  # Environment variables
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Set up pre-commit hooks:
```bash
pre-commit install
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Neo4j](https://neo4j.com/) for the graph database
- [SpaCy](https://spacy.io/) for NLP capabilities
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Hugging Face Transformers](https://huggingface.co/transformers/) for BERT models 