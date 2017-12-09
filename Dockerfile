FROM pytorch/pytorch:latest
RUN git clone https://github.com/mrdrozdov/mrdrozdov/translate-research.git && cd translate-research/python && pip install -r requirements.txt && python setup.py install