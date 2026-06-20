# SiliconFlow Model Prices

Fetch and display pricing for AI models on [SiliconFlow](https://siliconflow.cn/).

## Quick Start

```bash
pip install -r requirements.txt
python siliconflow_cn.py
```

## Output

```bash
python siliconflow_cn.py --md siliconflow_cn_prices.md --csv siliconflow_cn_prices.csv
```

| Org | Model ID | Type | Context | Unit | Cache | Input | Output |
|-----|----------|------|--------:|-----:|------:|------:|------:|
| deepseek-ai | DeepSeek-V3 | chat | 163.8K | 1M tokens | ¥0.20 | ¥2.00 | ¥8.00 |
| Qwen | Qwen3-8B | chat | 131.1K | 1M tokens | N/A | 免费 | 免费 |
| BAAI | bge-m3 | embedding | 8.2K | 1M tokens | N/A | 免费 | 免费 |
| Qwen | Qwen-Image | text-to-image | N/A | per image | N/A | ¥0.30 | N/A |

## Config

Edit `config.yaml` to set default output files and options.

## Troubleshooting

**SSL errors?** The script uses `truststore` for system certificates. If issues persist:

```bash
pip install --upgrade certifi
```

**Windows encoding issues?** The script handles UTF-8 automatically. If needed:

```bash
set PYTHONIOENCODING=utf-8
python siliconflow_cn.py
```
