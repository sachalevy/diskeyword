# DisKeyword

Source code for the `DisKeyword` demo paper. `DisKeyword` helps you search for topic-relevant hashtags within large tweet corpora.

## Quick Start

To get started directly and try out `DisKeyword`, use this [online demo](https://sachalevy-diskeyword-webstreamlit-app-xuzq6a.streamlitapp.com/) with sample datasets [uploaded here](https://drive.google.com/drive/folders/10eHg67mk3tKa5OrAS0nBVSVMXq7LhT7t?usp=sharing).

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

## Maintainers
- Sacha Lévy (sacha.levy@mail.mcgill.ca)