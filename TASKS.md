# QSweepy Plotting – worklog

## Основные проблемы (до правок)
- Dev-сервер Dash/Flask в одном процессе/потоке → блокировки при нескольких вкладках.
- Каждый callback открывает новое соединение к Postgres; нет пула, нет таймаутов.
- Конфиг и креды прошиты в коде; пути/порты завязаны на конкретную машину.
- Огромные таблицы/фигуры гонятся целиком в браузер; нет лимитов/кеша.
- Монолитный `vplot.py` без разделения обязанностей и без WSGI entrypoint.

## Что сделано
- Добавлен thread-safe пул подключений к Postgres (psycopg2 SimpleConnectionPool) и обёртки `run_sql_dataframe/get_conn`.
- Настройки БД/пула и каталог выгрузки SVG переехали в env переменные (см. `conf.py`).
- WSGI-ready: `server = app.server`, layout привязан при импорте; запуск через waitress (многопоточный) вместо dev-сервера по умолчанию.
- Переведено на `pyproject.toml` с зависимостями (теперь без `requirements.txt`).

## Как запускать сейчас
- Windows (prod-ish): `waitress-serve --listen=0.0.0.0:8060 vplot:server`
- Linux/WSL/Docker: `gunicorn --workers 2 --threads 4 vplot:server` (если gunicorn установлен), либо `waitress-serve` как выше.
- Локально для отладки: `python vplot.py` (упадёт обратно на встроенный сервер, если нет waitress).

Полезные env переменные:
- `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`
- `DB_POOL_MIN`, `DB_POOL_MAX`
- `SVG_OUTPUT_PATH`
- `WSGI_HOST`, `WSGI_PORT`, `WSGI_THREADS` (для запуска через `python vplot.py`)

## Следующие шаги (рекомендации)
- Разбить `vplot.py` на модули: layout, callbacks, db, plotting.
- Ввести лимиты/страничность для таблиц и downsampling для графиков; кеширование общих запросов.
- Обернуть работу с exdir/metadata в фоновые задачи или кеш, чтобы не блокировать UI.
- Добавить нормальный логгинг (файл/stdout, уровни) вместо `print`.
- Написать README с командами запуска и переменными окружения.
