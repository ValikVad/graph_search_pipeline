# GraphSearch

Этот репозиторий нужен для оценки пайплайна поиска по коду. Пайплайн состоит из трех шагов:

1. `Dual encoder` достает стартовый набор сниппетов.
2. `AST expansion` добавляет соседей по call graph.
3. `Cross encoder` заново ранжирует кандидатов.

Основные результаты лежат в `reports/`, `full_reports/` и `figures/`.

## Что здесь за что отвечает

### Основные скрипты

- [run_eval.py](/home/artemkhorev/edu/coirmodding/graphsearch/run_eval.py:1)  
  Главный скрипт для запуска оценки. Он:
  - при необходимости парсит репозитории;
  - прогоняет пайплайн на `questions/` и `questions_hard/`;
  - сохраняет агрегированные метрики в `reports/{repo}.json`;
  - сохраняет результаты по каждому вопросу в `full_reports/{repo}.json`.

- [evaluate.py](/home/artemkhorev/edu/coirmodding/graphsearch/evaluate.py:1)  
  Вся логика оценки и расчета метрик. Именно здесь считается, найден ли правильный сниппет на каждом этапе.

- [parse_ast.py](/home/artemkhorev/edu/coirmodding/graphsearch/parse_ast.py:1)  
  Режет репозиторий на сниппеты и строит call graph. Результат кладется в `ast_cache/{repo}/`.

- [run_ablation.py](/home/artemkhorev/edu/coirmodding/graphsearch/run_ablation.py:1)  
  Отдельный запуск для сравнения:
  - reranking без AST expansion;
  - reranking с AST expansion.  
  Результаты сохраняются в `ablation_reports/`.

- [search.py](/home/artemkhorev/edu/coirmodding/graphsearch/search.py:1)  
  Вспомогательный скрипт для ручного поиска по одному репозиторию.

### Папка `pipeline/`

- [pipeline/config.py](/home/artemkhorev/edu/coirmodding/graphsearch/pipeline/config.py:1)  
  Общие настройки: модели, `top_n`, `hop_depth`, `top_k`, batch size, device.

- [pipeline/index.py](/home/artemkhorev/edu/coirmodding/graphsearch/pipeline/index.py:1)  
  Загрузка сниппетов и call graph из `ast_cache/`.

- [pipeline/dual_encoder.py](/home/artemkhorev/edu/coirmodding/graphsearch/pipeline/dual_encoder.py:1)  
  Первый этап: поиск стартовых кандидатов.

- [pipeline/expander.py](/home/artemkhorev/edu/coirmodding/graphsearch/pipeline/expander.py:1)  
  Второй этап: расширение кандидатов соседями по графу вызовов.

- [pipeline/reranker.py](/home/artemkhorev/edu/coirmodding/graphsearch/pipeline/reranker.py:1)  
  Третий этап: финальный reranking cross-encoder’ом.

- [pipeline/pipeline.py](/home/artemkhorev/edu/coirmodding/graphsearch/pipeline/pipeline.py:1)  
  Обертка над всем пайплайном.

- [pipeline/visualize.py](/home/artemkhorev/edu/coirmodding/graphsearch/pipeline/visualize.py:1)  
  Визуализация neighborhood-графа для поиска.

## Данные

- `python_repos/`  
  Репозитории, по которым идет поиск.

- `questions/`  
  Более простой набор вопросов.

- `questions_hard/`  
  Более сложный набор вопросов.

## Что генерируется

- `ast_cache/`  
  Сниппеты и call graph для каждого репозитория после парсинга.

- `reports/`  
  Короткие агрегированные отчеты по каждому репозиторию.

- `full_reports/`  
  Подробные отчеты по каждому вопросу. Это главный источник для анализа и графиков.

- `ablation_reports/`  
  Отчеты для сравнения режимов с AST и без AST.

- `figures/`  
  Все сохраненные графики и схемы.

## Скрипты для графиков

- [plot_full_reports.py](/home/artemkhorev/edu/coirmodding/graphsearch/plot_full_reports.py:1)  
  Основной скрипт для построения графиков по `full_reports/`. Сейчас он генерирует:
  - `figures/full_reports/`
  - `figures/full_reports_stage1_top10/`
  - `figures/full_reports_stage1_top3/`

- [plot_repo_snippet_counts.py](/home/artemkhorev/edu/coirmodding/graphsearch/plot_repo_snippet_counts.py:1)  
  График с числом сниппетов по репозиториям.

- [plot_hard_top3_increase.py](/home/artemkhorev/edu/coirmodding/graphsearch/plot_hard_top3_increase.py:1)  
  График прироста `top-3` на hard-вопросах.

- [render_architecture_diagram.sh](/home/artemkhorev/edu/coirmodding/graphsearch/render_architecture_diagram.sh:1)  
  Пересобирает PNG со схемой архитектуры.

- [figures/architecture/graphsearch_architecture.html](/home/artemkhorev/edu/coirmodding/graphsearch/figures/architecture/graphsearch_architecture.html:1)  
  Исходник схемы архитектуры.

## Основные команды

Установить зависимости:

```bash
pip install -r requirements.txt
```

Запустить полную оценку:

```bash
python run_eval.py --skip-parse
```

Запустить ablation:

```bash
python run_ablation.py --repos browser-use_browser-use docling-project_docling
```

Пересобрать графики по отчетам:

```bash
python plot_full_reports.py
```

Пересобрать схему архитектуры:

```bash
bash render_architecture_diagram.sh
```

## Если коротко

Если нужен только основной рабочий контур проекта, смотри сюда:

- `run_eval.py`
- `evaluate.py`
- `parse_ast.py`
- `pipeline/`
- `questions/`
- `questions_hard/`
- `full_reports/`
- `plot_full_reports.py`

Остальное — либо кэш, либо уже сгенерированные результаты, либо вспомогательные скрипты.
