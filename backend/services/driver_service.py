"""
Driver service — aggregates driver statistics from raw_ride_orders
and populates/updates the drivers table.
"""
import logging

from sqlalchemy.orm import Session
from sqlalchemy import func, case

from backend.models.driver import Driver
from backend.models.raw_ride_order import RawRideOrder
from backend.models.engineered_ride_order import EngineeredRideOrder

logger = logging.getLogger(__name__)


def refresh_driver_stats(db: Session) -> dict:
    """
    Recompute all driver statistics from raw_ride_orders and update the drivers table.
    """
    # Aggregate stats from raw_ride_orders
    stats = (
        db.query(
            RawRideOrder.driver_id,
            func.count(RawRideOrder.id).label("total_rides"),
            func.sum(case((RawRideOrder.order_delayed == 1, 1), else_=0)).label("delayed_rides"),
            func.avg(RawRideOrder.total_delay_minutes).label("avg_delay"),
            func.avg(RawRideOrder.driver_base_rating).label("avg_rating"),
            func.max(RawRideOrder.booking_timestamp).label("last_ride_at"),
        )
        .group_by(RawRideOrder.driver_id)
        .all()
    )

    # Get reliability/risk/efficiency from engineered features (latest per driver)
    driver_scores = {}
    score_query = (
        db.query(
            RawRideOrder.driver_id,
            func.avg(EngineeredRideOrder.driver_reliability).label("reliability"),
            func.avg(EngineeredRideOrder.driver_risk_score).label("risk"),
            func.avg(EngineeredRideOrder.driver_efficiency_score).label("efficiency"),
        )
        .join(EngineeredRideOrder, EngineeredRideOrder.raw_order_id == RawRideOrder.id)
        .group_by(RawRideOrder.driver_id)
        .all()
    )
    for row in score_query:
        driver_scores[row.driver_id] = {
            "reliability": row.reliability or 0.0,
            "risk": row.risk or 0.0,
            "efficiency": row.efficiency or 0.0,
        }

    updated = 0
    for row in stats:
        total = row.total_rides or 0
        delayed = row.delayed_rides or 0
        delay_rate = delayed / total if total > 0 else 0.0
        scores = driver_scores.get(row.driver_id, {"reliability": 0.0, "risk": 0.0, "efficiency": 0.0})

        existing = db.query(Driver).filter(Driver.driver_id == row.driver_id).first()
        if existing:
            existing.total_rides = total
            existing.delayed_rides = delayed
            existing.delay_rate = round(delay_rate, 4)
            existing.avg_delay_minutes = round(row.avg_delay or 0.0, 2)
            existing.avg_rating = round(row.avg_rating or 0.0, 2)
            existing.reliability_score = round(scores["reliability"], 4)
            existing.risk_score = round(scores["risk"], 4)
            existing.efficiency_score = round(scores["efficiency"], 4)
        else:
            driver = Driver(
                driver_id=row.driver_id,
                total_rides=total,
                delayed_rides=delayed,
                delay_rate=round(delay_rate, 4),
                avg_delay_minutes=round(row.avg_delay or 0.0, 2),
                avg_rating=round(row.avg_rating or 0.0, 2),
                reliability_score=round(scores["reliability"], 4),
                risk_score=round(scores["risk"], 4),
                efficiency_score=round(scores["efficiency"], 4),
            )
            db.add(driver)
        updated += 1

    db.commit()
    logger.info(f"Refreshed stats for {updated} drivers")
    return {"message": f"Updated {updated} driver records", "drivers_updated": updated}
