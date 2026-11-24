"""
Étape 5 - Déploiement et planification des flows (piste d'excellence).

Terminal 1 : prefect server start
Terminal 2 : prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
             python deploiement_prefect.py
Terminal 3 : prefect deployment ls
             prefect deployment run 'all/ml-pipeline-all'
             prefect deployment run 'train/ml-pipeline-train'
"""

from prefect import serve
from prefect.client.schemas.schedules import CronSchedule

from pipeline_prefect import flow_all, flow_train, flow_evaluate, flow_code

TZ = "Africa/Tunis"

if __name__ == "__main__":
    serve(
        # Pipeline complet tous les jours à 02:00
        flow_all.to_deployment(
            name="ml-pipeline-all",
            schedules=[CronSchedule(cron="0 2 * * *", timezone=TZ)],
            tags=["mlops", "full-pipeline"],
        ),
        # Ré-entraînement tous les jours à 03:00
        flow_train.to_deployment(
            name="ml-pipeline-train",
            schedules=[CronSchedule(cron="0 3 * * *", timezone=TZ)],
            tags=["mlops", "train"],
        ),
        # Lancement manuel uniquement (pas de schedule)
        flow_evaluate.to_deployment(name="ml-pipeline-evaluate", tags=["mlops"]),
        flow_code.to_deployment(name="ml-pipeline-code", tags=["mlops", "code"]),
    )
