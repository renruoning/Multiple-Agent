"""模拟机票查询系统（真实项目中可替换为携程/航司/GDS 等 API）。"""
from __future__ import annotations

import random

AIRLINES = ["国航", "东航", "南航", "海南航空", "深圳航空", "厦门航空"]
AIRLINE_CODES = ["CA", "MU", "CZ", "HU", "ZH", "MF"]


def search_flights(
    origin: str,
    destination: str,
    date: str,
    cabin: str = "economy",
    n: int = 4,
) -> list[dict]:
    """查询指定航线、日期、舱位下的候选航班（确定性伪随机，便于测试）。"""
    rng = random.Random(f"{origin}-{destination}-{date}-{cabin}")
    base_price = rng.randint(600, 2200)

    options = []
    for i in range(n):
        idx = rng.randrange(len(AIRLINE_CODES))
        price = base_price + i * rng.randint(80, 300)
        if cabin == "business":
            price = int(price * 3)
        options.append(
            {
                "flight_no": f"{AIRLINE_CODES[idx]}{rng.randint(1000, 9999)}",
                "airline": AIRLINES[idx],
                "origin": origin,
                "destination": destination,
                "date": date,
                "depart_time": f"{rng.randint(6, 21):02d}:{rng.choice(['00', '15', '30', '45'])}",
                "cabin": cabin,
                "price": price,
            }
        )
    return sorted(options, key=lambda o: o["price"])
