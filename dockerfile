FROM python:3.8

RUN curl -fsSL https://starship.rs/install.sh > starship.sh
RUN bash starship.sh --yes
RUN echo 'eval "$(starship init bash)"' >> root/.bashrc
RUN rm starship.sh

RUN pip install -U black
RUN pip install -U pylint

RUN pip install sklearn

RUN pip install -U tqdm