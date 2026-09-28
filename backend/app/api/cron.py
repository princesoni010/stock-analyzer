
from fastapi import APIRouter, BackgroundTasks, status
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cron", tags=["cron"])


class DummyTask:
    def retry(self, *args, **kwargs):
        pass
    class MaxRetriesExceededError(Exception):
        pass
def run_daily_workflow():
    try:
        from app.workers.tasks import overnight_news_fetch, market_data_update, daily_screener, morning_report
        
        logger.info("Starting Daily Workflow (Cron) - News Fetch")
        overnight_news_fetch(DummyTask())
        logger.info("Starting Daily Workflow (Cron) - Market Data")
        market_data_update(DummyTask())
        
        logger.info("Starting Daily Workflow (Cron) - Screener")
        daily_screener(DummyTask())
        
        logger.info("Starting Daily Workflow (Cron) - Morning Report")
        morning_report(DummyTask())
        
        logger.info("Daily Workflow Completed!")
    except Exception as e:
        logger.error(f"Error in daily workflow: {e}", exc_info=True)


@router.get("/daily", status_code=status.HTTP_202_ACCEPTED)
@router.post("/daily", status_code=status.HTTP_202_ACCEPTED)
async def trigger_daily(background_tasks: BackgroundTasks):
    """
    Trigger the daily workflow (Fetch Data -> Run Screener -> Generate AI Report).
    Intended to be called by cron-job.org once a day at 7:58 AM IST.
    Runs in the background so the HTTP request returns immediately.
    """
    background_tasks.add_task(run_daily_workflow)
    return {"status": "accepted", "message": "Daily workflow started in background"}

