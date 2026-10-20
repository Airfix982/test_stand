NearS / SuperSpinner TimePack 20 kHz + STFT SIGNFLIP
===================================================

НАЗНАЧЕНИЕ
==========
Интеграция архива git_core.tar.gz с УЖЕ установленным в ~/nears/test_stand
TimePack 20k (sync/v2_timepack.py), универсальным crypto/frame_port.py
и экспериментальной маршрутизацией Chrome (experiment/telemostTest.py).

Новое имя: experiment.spectralTimepack20k
Новый адаптер: crypto.spectral_timepack_adaptor.SpectralTimepackAdaptor

ИЗМЕНЕНИЯ
=========
Скопированы ПОБАЙТОВО неизменные исходники:
    spectral_core/crypto/base.py
    spectral_core/crypto/stft.py
    spectral_core/crypto/mdct.py
    spectral_core/crypto/__init__.py

Сохранены ВСЕ режимы: stft/mdct x invert/shuffle/signflip; fixed/per_frame.
Все преобразования, RNG, маски/перестановки и хэш-ключи выполняются
ОРИГИНАЛЬНЫМИ методами из git_core. Никакого ALG7/HMAC от другого проекта.

Исключены из архива старые app.py, robust_sync.py, qim_sync.py, sync.py,
audio_sync.py, VirtualDevices.py и прочие старые синхронизаторы.
Они НЕ нужны при использовании SuperSpinner и не заменяют существующий код.

crypto/frame_port.py включён в архив без изменения как API-зависимость.
experiment/cryptoTimepack20k.py и ALG7 не изменяются.
sync/v2_timepack.py и experiment/telemostTest.py не заменяются.

ВАЖНО: FRAME INDEX
==================
TimePack кадр: 320 payload @ 16kHz + 80 guard @ 20kHz; 400 samples/20ms.
STFT окно: 2048 samples @16kHz, STFT hop: 1024 samples.
Поэтому индекс кадров STFT НЕ РАВЕН индексу TimePack/PN.
Адаптер инкрементирует STFT index только силами original STFTCipher.

Для каждого кадра, потерянного синхронизатором, адаптер подставляет
320 нулевых сэмплов В ШИФРОТЕКСТ (до дешифратора), сохраняя смещение окон
и последовательность исходных per_frame-ключей.
После пропуска восстановление речи может занимать несколько STFT-окон.

При старте, как в исходном app.py, отбрасывается один warmup STFT hop.
Чтобы сохранить непрерывный транспорт, используется буфер с постоянной
задержкой. При стандартном STFT delay=10 транспортных кадров на TX и
ещё 10 на RX (плюс задержки микрофона, сети и т.д.).
Это около 400ms только на 2 буфера; пока ЭКСПЕРИМЕНТАЛЬНО.

Используется ИСХОДНЫЙ алгоритм; качество восстановления после Opus
не гарантировано. Синхронизация может быть идеальной при плохой речи.

УСЛОВИЯ
=======
В каталоге test_stand должны уже существовать:
    sync/v2_timepack.py
    experiment/telemostTest.py
    crypto/frame_port.py (в архиве есть его неизменная копия)

Установка: в корне проекта
    cd ~/nears/test_stand && unzip -o ~/Загрузки/ECO_STFT_SIGNFLIP_TIMEPACK_V1.zip -d .

КЛЮЧ
====
--key-file содержит БИНАРНЫЕ байты, как исходный git_core/key.bin,
а НЕ hex-строку. TX и RX должны читать идентичный файл.
Использовать старый raw key.bin можно, передав полный путь.
Для локального ТЕСТА создать свой ключ:
    cd ~/nears/test_stand && mkdir -p .secrets && python3 -c 'import secrets,pathlib; p=pathlib.Path(".secrets/stft_test.key"); p.write_bytes(secrets.token_bytes(32)); p.chmod(0o600); print(p)'
НЕ присылайте реальный ключ третьим лицам и не коммитьте .secrets.

1. АВТОТЕСТ АДАПТЕРА (без TimePack и Telemost)
================================================
    cd ~/nears/test_stand && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m analysis.testSpectralTimepack

2. ШИФРОВАНИЕ WAV ЛОКАЛЬНО
=========================
Замените путь /path/to/input.wav на ваш WAV с речью.
    cd ~/nears/test_stand && python3 -m experiment.spectralTimepack20k --role tx --transform stft --operation signflip --key-schedule per_frame --key-file .secrets/stft_test.key --input /path/to/input.wav --generate-only --out /tmp/stft_tp_tx

3. ПРИЁМ WAV ЛОКАЛЬНО
====================
    cd ~/nears/test_stand && python3 -m experiment.spectralTimepack20k --role rx --transform stft --operation signflip --key-schedule per_frame --key-file .secrets/stft_test.key --wav /tmp/stft_tp_tx/tx_encrypted_20k.wav --out /tmp/stft_tp_rx

Проверка:
    cat /tmp/stft_tp_rx/summary.json
    paplay /tmp/stft_tp_rx/rx_decrypted_16k.wav

Ожидается initial acquired, frames_lost=0, последний frameIndex=N-1.
ВАЖНО: это тест синхронизации, не гарантия качественной дешифровки.

4. ЧЕРЕЗ OPUS ЛОКАЛЬНО
=====================
    ffmpeg -y -i /tmp/stft_tp_tx/tx_encrypted_20k.wav -ar 48000 -ac 1 -c:a libopus -application voip -b:a 48k /tmp/stft_tp.opus
    ffmpeg -y -i /tmp/stft_tp.opus -ar 20000 -ac 1 -c:a pcm_s16le /tmp/stft_tp_after_opus.wav
    cd ~/nears/test_stand && python3 -m experiment.spectralTimepack20k --role rx --transform stft --operation signflip --key-schedule per_frame --key-file .secrets/stft_test.key --wav /tmp/stft_tp_after_opus.wav --out /tmp/stft_tp_opus_rx

5. ЧЕРЕЗ TELEMOST
=================
Для Telemost настройте одинаковые ключ и --session на обеих машинах.
RX сначала:
    cd ~/nears/test_stand && python3 -m experiment.spectralTimepack20k --role rx --transform stft --operation signflip --key-file .secrets/stft_test.key --out /tmp/stft_live_rx --listen
В Telemost выбрать динамики NearS_RX, нажать Enter.
TX затем (микрофон):
    cd ~/nears/test_stand && python3 -m experiment.spectralTimepack20k --role tx --transform stft --operation signflip --key-file .secrets/stft_test.key --out /tmp/stft_live_tx --duration 30
В Telemost выбрать микрофон NearS_TX_Microphone, нажать Enter.
AutoRouter автоматически поправит соединения Chrome/PulseAudio.

СМЕНА РЕЖИМА
============
--transform stft|mdct
--operation signflip|shuffle|invert
--key-schedule per_frame|fixed
--session <одинаковая строка на TX/RX>
--stft-frame 2048 --stft-hop 1024 (исходные значения)
--mdct-half 2500 (исходное значение)

ВЫЯВЛЕННЫЕ ОГРАНИЧЕНИЯ
======================
* Начальная эпоха TimePack должна быть согласована: PN повторяется по модулю64.
  Подключение посреди разговора не устанавливает абсолютный STFT frame index.
* После длинных пропусков TimePack потерянный PCM не восстановить: вставляются нули.
* STFT с перекрытием не гарантирует точной обратимости после второго STFT;
  signflip не превращает Opus в безошибочный канал. Оценивайте реальное аудио.
* Тестовый API синхронизатора работает на 320 payload samples, но не знает
  STFT hop/keys; ими полностью управляет spectral_timepack_adaptor.py.
* Старый git_core/key.bin НЕ добавлен в пакет — материал ключа не распространяем.
