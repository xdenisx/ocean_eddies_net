# Сегментация океанских вихрей

Пакет содержит предобработку, обучение, инференс, постобработку, проверку качества на основе
GeoTIFF-изображений. Поддерживается **попиксельная семантическая сегментация**:
один класс на пиксель, включая фон.

**Версия 0.1.0rc1** 

## 1. Установка

Распакуйте архив и откройте терминал в корне репозитория.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Установите подходящую вашей системе сборку PyTorch/torchvision, затем:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
eddy-doctor
```

Для DeepLabV3+, U-Net++ и SegFormer достаточно `pip install -e ".[smp]"`.
Без внешних архитектур: `pip install -e .`. Для тестов: `pip install -e ".[dev]"`.
Рекомендуется создать отдельное окружение для работы с пакетом.
В Jupyter используйте `%pip` в нужном ядре и перезапустите его после установки.
Диагностика `eddy-doctor` выводит путь интерпретатора и состояние зависимостей.

## 2. Изображения и маски

```text
data/train/images/scene_001.tif
data/train/masks/scene_001.tif
data/test/images/scene_101.tif
data/test/masks/scene_101.tif
```

Совпадать должны относительные пути, имена без расширения, размер, CRS и transform.
Маски одноканальные, с целочисленными значениями. Изображения могут иметь несколько каналов: задайте их количество через параметр `in_channels`. Порядок каналов должен совпадать на всех этапах.

Бинарная схема масок: **0 — фон, 255 — вихрь**.
Для нескольких классов задайте таблицу в YAML:

```yaml
in_channels: 1
class_values: [0, 100, 255]
class_names: [background, eddy_type_1, eddy_type_2]
ignore_values: [65535]
unknown_mask_policy: error
```

Замените имена и значения на свою разметку. Первый класс — фон. Внутри сети коды переводятся
в индексы `[0, 1, 2]`; в GeoTIFF записываются исходные `[0, 100, 255]`.
Поддерживается и обычная схема `[0, 1, 2]`, и большее число классов.
Число выходов модели автоматически равно длине `class_values`.

**255 не считается NoData, если это код класса.** Новые маски предсказаний имеют
тип `uint16`, значение NoData = 65535 и внутреннюю маску валидности. Нули и 255
остаются допустимыми классами. Неизвестные значения по умолчанию вызывают ошибку.
Для воспроизведения бинарной сегментации типа «остальное — фон» можно задать
`unknown_mask_policy: background`; для неразмеченных областей — `ignore`.

NoData, NaN, невалидные пиксели входных каналов и края тайлов исключены
из функций потерь и метрик. Полностью пустые тайлы не анализируются.
При конфликте объявленного NoData маски с явным кодом класса приоритет отдаетя таблице
классов.

## 3. Обучение

Готовые конфигурации:

```bash
eddy-train --config configs/binary_deeplab.yaml
eddy-train --config configs/multiclass_deeplab.yaml
```

Сначала отредактируйте пути и легенду классов. Например, DeepLabV3+ с одним
изображением в батче:

```bash
eddy-train --config configs/multiclass_deeplab.yaml \
  --output_dir runs/deeplab_multiclass_trial \
  --batch_size 1 --freeze_batchnorm --augment_level heavy
```

В PowerShell удобнее использовать YAML и одну строку запуска, без bash-символа `\`.

Сохраняются архитектуры DeepLabV3+ ResNet50/101, U-Net++ EfficientNet-B4/B5,
SegFormer B0/B2/B4/B5, UPerNet Swin-T/S и HRNet W18/W32.
`transunet` —  простая TransUNet сеть.
`sam_vit_unet` — усложненная TransUNet-подобная сеть,

Каждая архитектура получает нужное число выходных классов. Backend явно сохраняется
в checkpoint; другая сеть не подставляется незаметно при отсутствии библиотеки.
В новой реализации Hugging Face `pretrained: true` загружает веса, а не только config.

Размер тайлов по-умолчанию 512×512. Для аугментации доступны три степени ее применения `light`, `medium`, `heavy` и `--no_augment`.

Набор разбивается **по сценам до нарезки**. Для зависимых снимков задайте свой
`split_file` с группировкой по району/сроку: простое разделение по именам не гарантирует
независимости перекрывающихся сцен. Нужны минимум две пригодные сцены.
`split_mode: tile` оставлен только как явно включаемый вариант с предупреждением
об утечке. Положительные плитки можно фильтровать при обучении, но validation
всегда сохраняет фоновые плитки. По умолчанию `min_positive_fraction: 0`.

Функции потерь обобщены на несколько классов:
`ce`, `dice`, `dice_ce`, `focal`, `dice_focal`, `tversky`, `focal_tversky`.
`class_weights` задаются в порядке `class_values`. Скалярный `focal_alpha`
действует только в бинарном случае. Dice/Tversky по умолчанию усредняются без фона.

Early stopping: `patience`, `early_stop_monitor` (`val_iou`, `val_dice`, `val_loss`),
`min_delta`. Для нескольких классов `val_iou`/`val_dice` — среднее по определённым
метрикам нефоновых классов. Лучший checkpoint сохраняется при любом строгом
улучшении; `min_delta` управляет только сбросом счётчика patience.

```text
best_checkpoint.pt  last_checkpoint.pt
history.json        training_log.csv
config.json         class_schema.json
split.json          environment.json
```

`--freeze_batchnorm` замораживает текущие статистики BatchNorm после каждого
`model.train()`; обучаемые scale/bias остаются. `drop_last: true` действует только
для обучения. Accumulation не увеличивает реальный батч для BatchNorm.
Для продолжения прерванного запуска используйте тот же config и
`--resume runs/имя/last_checkpoint.pt`; не меняйте число плановых эпох и параметры.

## 4. Инференс и постобработка

```bash
eddy-infer --input data/test/images \
  --checkpoint runs/deeplab_multiclass/best_checkpoint.pt \
  --output_dir predictions/multiclass \
  --tile_size 512 --stride 256 --blend_mode distance \
  --smooth_sigma 1.0 --min_object_size 200 --min_hole_size 100 \
  --save_prob --save_raw_prob
```

Архитектура, классы, каналы и нормализация берутся из checkpoint.
Веса загружаются строго; неполная загрузка не маскируется предупреждением.
Весовые окна: `uniform`, `distance`, `hann`, `gaussian`.
Накопление вероятностей всех классов — на диске; `--work_dir` задаёт временный каталог.
При этом память всё равно нужна для целых двумерных карт и связных компонент.

Для бинарной сети можно задать `--threshold 0.35`. Для нескольких классов
используется `argmax`; независимые бинарные пороги не создают конфликтующих меток.
Сглаживаются вероятности, а не числовые коды классов. Малые объекты удаляются
поклассово, дырки не заполняются поверх другого класса или NoData.
Все размеры в пикселях. Для разных классов доступны разные пороги размера:
`--class_min_sizes '{"100":200,"255":100}'`.
Настраивайте пороги на validation, а не на итоговом test.

Выходы: `scene_pred.tif`, `scene_pred_prob.tif`, `scene_pred_raw_prob.tif`.
Вероятностные GeoTIFF содержат C каналов, включая фон, в порядке легенды.
`_prob` — после сглаживания, до морфологической обработки; `_raw_prob` —
взвешенное объединение до сглаживания. NoData вероятностей — NaN.

## 5. Проверка качества

```bash
eddy-validate --pred_dir predictions/multiclass \
  --mask_dir data/test/masks --images_dir data/test/images \
  --checkpoint runs/deeplab_multiclass/best_checkpoint.pt \
  --output_dir metrics/multiclass
```

Подсчитываются IoU, Dice/F1, precision, recall, specificity, accuracy, TP/FP/FN/TN,
поклассовые и средние показатели; confusion matrix складывается по всем пригодным
пикселям, а не усредняется из отношений батчей. Строки матрицы — эталон,
столбцы — предсказание. Неопределённые метрики записываются как `null`, не как 1.
Пропуски и исключённые области отражены в отчёте. Это попиксельные метрики;
метрики отдельных вихрей/экземпляров в этой версии не реализованы.

## 6. HistEq + CLAHE

```bash
eddy-preprocess --input_dir data/train/images \
  --output_dir data/train/images_hist_clahe --clip_limit 0.03
```

Цепочка: `cv2.normalize(0..1)` → `equalize_hist` → `equalize_adapthist` →
`cv2.normalize(0..255, uint8)`. Имена и геопривязка сохраняются, каналы обрабатываются
отдельно. `--skip_equalize_hist` оставляет CLAHE без глобальной эквализации.
`--kernel_size 128` задаёт размер области; 0 — автоматический выбор skimage.
Сами маски классов предобрабатывать не нужно.

CLAHE не принимает маску валидности: временное заполнение ближайшими валидными
значениями — приближение, после которого NoData восстанавливается.
Для таких выходных изображений укажите **`zero_is_nodata: false`**: 0 теперь может
быть валидной яркостью; валидность хранится отдельной GDAL-маской. Настройка
попадает в checkpoint. Обрабатывайте train/validation/test одинаково.

## 7. Старые модели, тесты и публикация

```bash
eddy-convert-legacy --input old_run/best_checkpoint.pt --output converted_binary.pt
```

Конвертер проверяет соответствие весов, сохраняет бинарную легенду и исходную
формулу нормализации. Это не обучение дополнительных классов. Для старого HF
может понадобиться загрузка конфигурации один раз; у нового checkpoint она внутри.
Совместимость всех старых внешних backend-версий не гарантирована — при конфликте
есть неизменённые исходные скрипты. Многоклассовую сеть нужно обучить с новой разметкой.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/make_demo_data.py
eddy-train --config configs/smoke_cpu.yaml
```

Фактические проверки и ограничения перечислены в `docs/TEST_REPORT.md`.
Синтетические проверки не оценивают качество на ваших вихрях.

Для GitHub добавлены `pyproject.toml`, `.gitignore`, `.gitattributes`, тесты,
CI workflow и документация. Инструкция `git init/add/commit/push` — в README.md.
Архив сам ничего не публикует. **Лицензия и правообладатель не назначены**:
перед публичным размещением проверьте `NOTICE_RELEASE.md`, укажите согласованную
лицензию и авторство. Данные и веса по умолчанию исключены из Git.
