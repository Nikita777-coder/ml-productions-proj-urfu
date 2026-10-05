# FashionMNIST production pipeline

Рефакторинг [Colab-ноутбука](https://colab.research.google.com/drive/1m-WKcpdE7qCQRipzM4ICgtDRQEpK0xPM) под production-стандарты: пакет, конфиг, CLI, тесты, Poetry, линтеры и pre-commit.

Исходный скрипт обучал CNN на FashionMNIST в одной ячейке. Здесь тот же алгоритм разложен по модулям и запускается воспроизводимо из Git-репозитория.

## Структура

```
src/fashion_mnist/   # production-код
  config.py          # гиперпараметры и пути
  data.py            # датасеты и DataLoader
  model.py           # FashionCNN
  train.py           # обучение и сохранение чекпоинта
  evaluate.py        # оценка на тесте
  device.py          # CPU / CUDA / MPS и seed
tests/               # быстрые unit-тесты без полного обучения
notebooks/           # ссылка на исходный Colab
```

## Виртуальное окружение в Git

Окружение создаётся **внутри репозитория** (`.venv`) через Poetry:

- `poetry.toml` включает `virtualenvs.in-project = true`
- `.python-version` фиксирует Python 3.12
- `poetry.lock` фиксирует точные версии зависимостей
- сам каталог `.venv` в `.gitignore`: в Git хранится способ воспроизвести окружение

```bash
python3 -m pip install --user poetry
poetry install
poetry env info
```

После `poetry install` в корне появится `.venv`, привязанный к этому репозиторию.

## Команды

```bash
poetry run pre-commit install
poetry run pytest
poetry run ruff check src tests
poetry run black --check src tests
poetry run train --epochs 5
poetry run evaluate
```

Чекпоинт сохраняется в `artifacts/fashion_cnn.pt`. Датасет качается в `data/`.

Если появится `RuntimeError: File not found or corrupted`, удалите оборванную загрузку и скачайте заново:

```bash
rm -rf data/FashionMNIST
poetry run train --epochs 1
```

Код сам качает датасет с Google Storage / GitHub и проверяет IDX-файлы. Стандартный URL torchvision (`fashion-mnist.s3-website`) часто обрывается на 99% и оставляет битый архив.
