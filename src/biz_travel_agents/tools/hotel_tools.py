"""模拟酒店查询系统（真实项目中可替换为携程/Booking 等 API）。"""
from __future__ import annotations

import random
from datetime import date, timedelta

HOTEL_BRANDS = ["如家", "汉庭", "全季", "希尔顿", "万豪", "香格里拉", "锦江之星"]


def _nights(checkin: str, checkout: str) -> int:
    try:
        d1 = date.fromisoformat(checkin)
        d2 = date.fromisoformat(checkout)
        return max((d2 - d1).days, 1)
    except ValueError:
        return 1


def search_hotels(
    city: str,
    checkin: str,
    checkout: str,
    n: int = 4,
) -> list[dict]:
    """查询指定城市、入离日期下的候选酒店（确定性伪随机，便于测试）。"""
    rng = random.Random(f"{city}-{checkin}-{checkout}")
    nights = _nights(checkin, checkout)
    base_price = rng.randint(280, 900)

    options = []
    for i in range(n):
        price_per_night = base_price + i * rng.randint(60, 220)
        options.append(
            {
                "hotel_name": f"{city}{rng.choice(HOTEL_BRANDS)}酒店",
                "city": city,
                "checkin": checkin,
                "checkout": checkout,
                "nights": nights,
                "star_rating": min(3 + i, 5),
                "price_per_night": price_per_night,
                "total_price": price_per_night * nights,
            }
        )
    return sorted(options, key=lambda o: o["price_per_night"])
