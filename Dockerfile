FROM python:3.9.13-slim


RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*


ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python - --version $POETRY_VERSION
ENV PATH="/root/.local/bin:$PATH"

COPY . /app
WORKDIR /app

RUN poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-ansi --only main


RUN chmod +x user_id_gender_prediction.sh

CMD ["./user_id_gender_prediction.sh"]