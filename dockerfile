FROM autogoal/autogoal:latest as autogoal

WORKDIR /code
RUN poetry build

FROM python:3.8

RUN curl -fsSL https://starship.rs/install.sh > starship.sh
RUN bash starship.sh --yes
RUN echo 'eval "$(starship init bash)"' >> root/.bashrc
RUN rm starship.sh

RUN pip install -U black
RUN pip install -U pylint

RUN pip install sklearn

RUN pip install -U tqdm

COPY --from=autogoal /code/dist/autogoal-0.1.0.tar.gz /autogoal-0.1.0.tar.gz
RUN pip install /autogoal-0.1.0.tar.gz
