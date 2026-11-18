"""Static и continuous batching без модели и GPU.

Запрос здесь — только число шагов генерации. Один шаг — один токен.
Static batching держит слот до конца самого длинного запроса в пачке.
Continuous batching освобождает слот сразу после EOS и ставит туда
следующий запрос из очереди.

Симуляторы меняют ``done_steps`` у переданных объектов. Для каждого
прогона нужен новый список запросов.
"""

from __future__ import annotations

from dataclasses import dataclass

DEMO_LENGTHS = (3, 8, 2, 5, 4, 3, 6, 2, 5, 4, 3, 7)


@dataclass
class Request:
    """Запрос на генерацию: идентификатор и длина ответа в шагах."""

    id: int
    total_steps: int
    done_steps: int = 0

    def step(self) -> None:
        """Один шаг генерации — один выданный токен."""
        self.done_steps += 1

    def is_done(self) -> bool:
        """Генерация закончилась, когда выданы все заданные токены."""
        return self.done_steps >= self.total_steps


def make_requests(lengths: list[int] | tuple[int, ...] = DEMO_LENGTHS) -> list[Request]:
    """Собирает очередь: позиция в списке становится id запроса."""
    if any(length < 1 for length in lengths):
        raise ValueError("длина ответа должна быть не меньше 1")
    return [Request(id=index, total_steps=length) for index, length in enumerate(lengths)]


def fresh_uniform(count: int = 8, steps: int = 100) -> list[Request]:
    """Одинаковые запросы: разброса длин нет."""
    return [Request(id=index, total_steps=steps) for index in range(count)]


def fresh_skewed() -> list[Request]:
    """Семь коротких по 25 шагов и один длинный на 625. Сумма та же, 800."""
    return [Request(id=index, total_steps=25) for index in range(7)] + [
        Request(id=7, total_steps=625)
    ]


def fresh_uniform_long() -> list[Request]:
    """32 одинаковых запроса по 100 шагов. Очередь длиннее батча из 8."""
    return fresh_uniform(32, 100)


def fresh_skewed_long() -> list[Request]:
    """28 коротких по 25 шагов, затем 4 длинных по 625. Длинные идут пачкой."""
    return [Request(id=index, total_steps=25) for index in range(28)] + [
        Request(id=28 + index, total_steps=625) for index in range(4)
    ]


def fresh_skewed_interleaved() -> list[Request]:
    """4 длинных запроса равномерно стоят среди 28 коротких.

    Длинный запрос идёт первым, затем короткий. После каждых семи
    коротких добавляется следующий длинный, пока их не станет четыре.
    """
    longs = [Request(id=100 + index, total_steps=625) for index in range(4)]
    shorts = [Request(id=index, total_steps=25) for index in range(28)]
    mixed = [longs[0]]
    for index, short in enumerate(shorts):
        mixed.append(short)
        long_index = (index + 1) // 7
        if (index + 1) % 7 == 0 and long_index < 4:
            mixed.append(longs[long_index])
    return mixed


def static_batch_waste(lengths: list[int]) -> dict[str, float | int]:
    """Считает пустые слото-шаги static batching, если все запросы в одном батче.

    Батч живёт ``max(lengths)`` шагов и на каждом шаге занимает все слоты.
    Полезная работа — сумма длин. Остальное — слоты уже завершившихся запросов.

    Args:
        lengths: Длины ответов в токенах, по одной на запрос.

    Returns:
        Полезные и полные слото-шаги, простой и его доля.
    """
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("нужна хотя бы одна положительная длина")
    longest = max(lengths)
    useful = sum(lengths)
    total = len(lengths) * longest
    wasted = total - useful
    return {
        "useful_slot_steps": useful,
        "total_slot_steps": total,
        "wasted_slot_steps": wasted,
        "waste_share": wasted / total,
    }


def _empty_result() -> dict:
    return {
        "wall_steps": 0,
        "useful_slot_steps": 0,
        "total_slot_steps": 0,
        "utilization": 0.0,
        "timeline": [],
    }


def simulate_static_batching(requests: list[Request], batch_size: int) -> dict:
    """Батч собирается один раз и живёт до конца самого длинного запроса.

    Завершившийся запрос остаётся в слоте и больше не делает полезной
    работы. Пустые места, если запросов меньше ``batch_size``, тоже
    считаются простоем.

    Args:
        requests: Очередь. Объекты будут изменены вызовами ``step``.
        batch_size: Сколько слотов обрабатывается одновременно.

    Returns:
        Число шагов по времени, полезные и все слото-шаги, утилизация
        и снимок id в каждом слоте на каждом шаге. ``None`` — простой.
    """
    if batch_size < 1:
        raise ValueError("batch_size должен быть не меньше 1")
    if not requests:
        return _empty_result()

    queue = list(requests)
    useful_slot_steps = 0
    total_slot_steps = 0
    wall_steps = 0
    timeline = []

    while queue:
        batch = [queue.pop(0) for _ in range(min(batch_size, len(queue)))]
        while any(not request.is_done() for request in batch):
            snapshot = []
            for request in batch:
                if not request.is_done():
                    request.step()
                    useful_slot_steps += 1
                    snapshot.append(request.id)
                else:
                    snapshot.append(None)
                total_slot_steps += 1
            while len(snapshot) < batch_size:
                snapshot.append(None)
                total_slot_steps += 1
            timeline.append(snapshot)
            wall_steps += 1

    return {
        "wall_steps": wall_steps,
        "useful_slot_steps": useful_slot_steps,
        "total_slot_steps": total_slot_steps,
        "utilization": useful_slot_steps / total_slot_steps,
        "timeline": timeline,
    }


def simulate_continuous_batching(requests: list[Request], batch_size: int) -> dict:
    """Освободившийся слот сразу занимает следующий запрос из очереди.

    Если очередь уже пуста, слот остаётся пустым до конца самой длинной
    генерации. Поэтому выигрыш есть только когда за батчем ещё кто-то стоит
    и короткие запросы приходят вперемешку с длинными.

    Args:
        requests: Очередь. Объекты будут изменены вызовами ``step``.
        batch_size: Сколько слотов держать занятыми.

    Returns:
        Те же поля, что у ``simulate_static_batching``.
    """
    if batch_size < 1:
        raise ValueError("batch_size должен быть не меньше 1")
    if not requests:
        return _empty_result()

    queue = list(requests)
    slots: list[Request | None] = [None] * batch_size
    useful_slot_steps = 0
    total_slot_steps = 0
    wall_steps = 0
    timeline = []

    for index in range(batch_size):
        if queue:
            slots[index] = queue.pop(0)

    while any(slot is not None for slot in slots):
        snapshot = []
        for request in slots:
            if request is not None:
                request.step()
                useful_slot_steps += 1
                total_slot_steps += 1
                snapshot.append(request.id)
            else:
                total_slot_steps += 1
                snapshot.append(None)
        timeline.append(snapshot)
        wall_steps += 1

        for index in range(batch_size):
            if slots[index] is not None and slots[index].is_done():
                slots[index] = None
        for index in range(batch_size):
            if slots[index] is None and queue:
                slots[index] = queue.pop(0)

    return {
        "wall_steps": wall_steps,
        "useful_slot_steps": useful_slot_steps,
        "total_slot_steps": total_slot_steps,
        "utilization": useful_slot_steps / total_slot_steps if total_slot_steps else 0.0,
        "timeline": timeline,
    }


def format_timeline(timeline: list[list[int | None]], max_steps: int = 50) -> str:
    """Рисует таблицу «слот × шаг»: буква — id запроса, точка — простой.

    Буква считается как ``A + id % 26``. На демонстрационных id 0…11
    это A…L без повторов.
    """
    if not timeline:
        return ""
    slot_count = max(len(step) for step in timeline)
    lines = []
    shown = min(len(timeline), max_steps)
    for slot_index in range(slot_count):
        cells = []
        for step in range(shown):
            value = timeline[step][slot_index] if slot_index < len(timeline[step]) else None
            cells.append("·" if value is None else chr(ord("A") + value % 26))
        lines.append(f"Слот {slot_index}: " + "".join(cells))
    return "\n".join(lines)


def compare_batching(requests_factory, batch_size: int) -> dict:
    """Прогоняет одну и ту же очередь обеими стратегиями.

    ``requests_factory`` вызывается дважды: симулятор меняет запросы,
    и второй прогон на тех же объектах увидел бы уже законченную работу.

    Args:
        requests_factory: Функция без аргументов, которая возвращает очередь.
        batch_size: Размер батча для обоих прогонов.

    Returns:
        Результаты static и continuous и отношение их ``wall_steps``.
    """
    static = simulate_static_batching(requests_factory(), batch_size)
    continuous = simulate_continuous_batching(requests_factory(), batch_size)
    return {
        "static": static,
        "continuous": continuous,
        "speedup": static["wall_steps"] / continuous["wall_steps"],
    }
