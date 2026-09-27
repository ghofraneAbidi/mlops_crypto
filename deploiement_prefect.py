from prefect import serve
from prefect.schedules import Cron

from pipeline_prefect import flow_all, flow_train


if __name__ == "__main__":
    deployment_all = flow_all.to_deployment(
        name="ml-pipeline-all",
        schedule=Cron(
            "0 2 * * *",
            timezone="Africa/Tunis",
        ),
    )

    deployment_train = flow_train.to_deployment(
        name="ml-pipeline-train",
        schedule=Cron(
            "0 3 * * *",
            timezone="Africa/Tunis",
        ),
    )

    serve(
        deployment_all,
        deployment_train,
    )
