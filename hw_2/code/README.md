# Воспроизведение экспериментов

Все команды выполняются из корня `hw_2`. Нужен Python 3.12 и PyTorch с CUDA для обучения на GPU; версии использованной среды записаны в [requirements.txt](../requirements.txt).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt
.venv/bin/python code/check_solution.py
.venv/bin/python code/check_optimizations.py
```

MNIST и CIFAR-10 автоматически загружаются в `data/` при первом обращении.

Все 28 конфигураций заданы в `run_experiments.py`; фактически использованные параметры сохранены в `runs/*/history.json` и `range_test.json`. Завершённые результаты пропускаются при повторном запуске:

```bash
bash code/run_all.sh all --gpus 0 1
.venv/bin/python code/verify_report2.py
```

Для полного повторения обучения запустите эту команду в отдельной копии комплекта без сохранённого каталога `runs`.

Отдельный новый запуск ResNet-20 CLR с тремя seed создаётся командами для seed 0, 1 и 2; пример для seed 0:

```bash
.venv/bin/python code/train.py --dataset cifar10 --batch-size 1000 --workers 2 --cache-data --weight-decay 0.0001 --bn-momentum 0.05 --momentum 0.9 --shortcut pad --eval-every 1000 --channels-last --model resnet20 --seed 0 --policy clr --min-lr 0.1 --max-lr 3 --stepsize 10000 --iters 20000 --out runs/reproduction/r20_clr_seed0
```