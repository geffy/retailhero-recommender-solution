#  Решение задачи [RetailHero.ai/#2](https://retailhero.ai/c/recommender_system/overview) [3-е место]

Собрано на основе [бэйслайна](https://github.com/datagym-ru/retailhero-recomender-baseline)

## Шаги по подготовке:

### Скопировать данные в data/raw
```
cd {REPO_ROOT}
mkdir -p data/raw
cp /path/to/upacked/data/*.csv ./data/raw
```


### Запусить основной скрипт
```bash
./main.sh
```

### Упаковать сабмит
```bash
cd submit
zip -r submit_title.zip solution/*
```

## Результаты: 
```
Check: 0,1403
Public: 0,1320
Private: 0,145728
```


## P.S.
В `submit/submit_v4.3f_noDaily_noDebug.zip` находится оригинальный файл сабмита. 

Данный репозиторий представляет более чистую (по сравнению с оригиналом) версию кода для сбора решения. Точная воспроизводимость результатов не гарантируется.

Версии библиотек перечислены в `requirements.txt`
