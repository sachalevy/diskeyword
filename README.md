# DisKeyword

Source code for the `DisKeyword` demo paper. `DisKeyword` helps you search for topic-relevant hashtags within large tweet corpora.

## Quick Start

To get started directly and try out `DisKeyword`, use this [online demo](https://diskeyword.streamlit.app) with sample datasets [uploaded here](https://drive.google.com/drive/folders/10eHg67mk3tKa5OrAS0nBVSVMXq7LhT7t?usp=sharing).

To run the demo locally, clone this project, and setup your python environment:
```bash
python3 -m venv env/
pip install -r requirements.txt
pip install -e .
```

Now, run the web application as follows:
```bash
streamlit run web/streamlit_app.py
```

The `DisKeyword` interface now pops up in your browser and is ready to use.

## BibTex Citation

Please cite the following paper if you use this work!

```
@inproceedings{10.1145/3539597.3573033,
author = {L\'{e}vy, Sacha and Rabbany, Reihaneh},
title = {DisKeyword: Tweet Corpora Exploration for Keyword Selection},
year = {2023},
isbn = {9781450394079},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3539597.3573033},
doi = {10.1145/3539597.3573033},
booktitle = {Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining},
pages = {1136–1139},
numpages = {4},
keywords = {keyword selection, web application, twitter dataset},
location = {Singapore, Singapore},
series = {WSDM '23}
}
```

## Maintainers

- Sacha Lévy (sacha.levy@mail.mcgill.ca)
