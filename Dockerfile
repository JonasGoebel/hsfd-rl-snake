FROM python:3.13
RUN pip3 install poetry
WORKDIR /code
COPY pyproject.toml .
RUN poetry config virtualenvs.create false
RUN poetry install
COPY snake snake
CMD ["poetry", "run", "python", "snake/snake.py"]
