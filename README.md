# CXR-LLaVA: Chest X-ray Large Language and Vision Assistant

CXR-LLaVA is an advanced AI system that combines large language models and computer vision to analyze chest X-ray images. Built on the LLaVA architecture, it provides detailed medical analysis and natural language descriptions of radiological findings.

## Features

- **Advanced X-ray Analysis**: Detailed analysis of chest X-rays using state-of-the-art vision-language models
- **Natural Language Reports**: Generate comprehensive reports in natural language
- **Medical Context**: Specialized in medical terminology and radiological findings
- **High Performance**: GPU-accelerated inference with caching
- **Secure**: JWT-based authentication and rate limiting
- **Batch Processing**: Support for analyzing multiple images asynchronously

## Prerequisites

- Python 3.8+
- PyTorch
- CUDA-capable GPU (recommended)
- 16GB+ RAM

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Greprovad-AI/Chest-X-Ray-Analyzer.git
cd Chest-X-Ray-Analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the application:
- Copy `config/api_config.yaml.example` to `config/api_config.yaml`
- Update the configuration values as needed

4. Start the server:
```bash
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## API Usage

### Authentication

```bash
# Get access token
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=your_username&password=your_password"
```

### Analyze Single Image

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Authorization: Bearer your_token" \
  -F "file=@path/to/xray.jpg" \
  -F "prompt=Analyze this chest X-ray and describe any findings in detail."
```

### Batch Analysis

```bash
curl -X POST "http://localhost:8000/batch/analyze" \
  -H "Authorization: Bearer your_token" \
  -F "files=@xray1.jpg" \
  -F "files=@xray2.jpg"
```

### Get Available Prompts

```bash
curl -X GET "http://localhost:8000/prompts" \
  -H "Authorization: Bearer your_token"
```

## Model Architecture

CXR-LLaVA is based on the LLaVA (Large Language and Vision Assistant) architecture, fine-tuned specifically for chest X-ray analysis. It combines:

- Vision Encoder: Pre-trained vision transformer for X-ray image understanding
- Language Model: Large language model specialized in medical terminology
- Cross-modal Fusion: Advanced attention mechanisms for combining visual and textual information

## Rate Limiting

- Single analysis: 10 requests/minute
- Batch analysis: 2 requests/minute

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use CXR-LLaVA in your research, please cite:

```bibtex
@misc{cxr-llava-2023,
  author = {Greprovad AI},
  title = {CXR-LLaVA: Chest X-ray Large Language and Vision Assistant},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Greprovad-AI/Chest-X-Ray-Analyzer}}
}
```

## Acknowledgments

- LLaVA framework
- PyTorch
- FastAPI
- Hugging Face Transformers
