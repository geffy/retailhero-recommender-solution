## Кастомный докер образ с поддержкой `pytorch 1.3` и `faiss`.

### Локальная сборка образа
```bash
docker build -t geffy/ds-base:retailhero ./
```
`geffy/ds-base:retailhero` является именем образа.

После этого образ можно использовать в локальных тестах. 

# Публикация контейнера
Для использования в проверяющей системе, образ должен быть выложен в публичный доступ. Сделать это можно командой (предварительно нужно залогиниться):

```bash
docker push geffy/ds-base:retailhero
```
