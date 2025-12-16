# main.py
# Vertex AI Pipelines entrypoint (no shell scripts needed)

import argparse
import datetime
from kfp import dsl, compiler
from kfp.dsl import Model, Dataset
import google.cloud.aiplatform as aip

# ====== ENV (edit these) ======
PROJECT_ID = "eng-reactor-287421"
REGION = "us-central1"
PIPELINE_ROOT = "gs://ficc-pipelines-test"
ENDPOINT_ID = ""  # should not be hardcoded
BASE_IMAGE = f"gcr.io/{PROJECT_ID}/training-base:test"  # same idea as before:contentReference[oaicite:3]{index=3}

# ====== helpers ======
def set_compute_resources(obj, cpu=None, mem=None, gpu=None, gpu_count=None):
    if cpu: obj.set_cpu_limit(cpu)
    if mem: obj.set_memory_limit(mem)
    if gpu: obj.add_node_selector_constraint(gpu)
    if gpu and gpu_count: obj.set_gpu_limit(gpu_count)
    if gpu and not gpu_count: obj.set_gpu_limit(1)

# ====== components (inline) ======
@dsl.component(
    base_image=BASE_IMAGE,
    target_image=f"gcr.io/{PROJECT_ID}/data_prep_component:test"
)
def data_prep_component(model: str,
                        testing: bool,
                        bucket_name: str = "ficc-pipelines-test") -> Dataset:
    import os, pandas as pd, pickle5 as pickle
    from datetime import datetime
    from automated_training.auxiliary_functions import save_update_data_results_to_pickle_files, update_data
    from automated_training.auxiliary_variables import WORKING_DIRECTORY

    # force aux funcs to use the test/prod toggles & bucket if needed
    # (kept minimal; your functions already read their own config)
    os.makedirs(f"{WORKING_DIRECTORY}/files", exist_ok=True)

    # Persist processed data & metadata (matches your current orchestrator):contentReference[oaicite:4]{index=4}
    data, last_trade_date, num_feats, raw_path = update_data(model)
    fp = f"{WORKING_DIRECTORY}/files/processed_data_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    data.to_pickle(fp)

    art = Dataset(uri=fp)
    art.metadata["training_data_path"] = fp
    art.metadata["last_trade_date"] = last_trade_date
    art.metadata["num_features_for_each_trade_in_history"] = num_feats
    art.metadata["raw_data_path"] = raw_path or ""
    return art


@dsl.component(
    base_image=BASE_IMAGE,
    target_image=f"gcr.io/{PROJECT_ID}/train_eval_component:test"
)
def train_eval_component(dataset: Dataset,
                         model: str,
                         testing: bool,
                         email_recipients: list,
                         learning_rate: float = 0.0001,
                         batch_size: int = 1000,
                         num_epochs: int = 100,
                         dropout: float = 0.1) -> Model:
    import os, json, pandas as pd
    from datetime import datetime
    import tensorflow as tf
    from automated_training.auxiliary_functions import (
        setup_gpus, train_model, get_trade_date_where_data_exists_on_this_date,
        decrement_business_days, get_model_results, send_no_new_model_email,
        send_results_email_multiple_tables
    )
    import automated_training.auxiliary_functions as ATAF
    from automated_training.auxiliary_variables import EASTERN, YEAR_MONTH_DAY

    # wire training hyperparams into your orchestrator (same pattern as before):contentReference[oaicite:5]{index=5}
    ATAF.LEARNING_RATE = learning_rate
    ATAF.BATCH_SIZE = batch_size
    ATAF.NUM_EPOCHS = num_epochs
    ATAF.DROPOUT = dropout
    ATAF.TESTING = testing

    setup_gpus()

    # load the data prepared in previous step (exactly like in your current KFP comp):contentReference[oaicite:6]{index=6}
    fp = dataset.metadata["training_data_path"]
    data = pd.read_pickle(fp)
    last_trade_date = dataset.metadata["last_trade_date"]
    n_hist_feats = dataset.metadata["num_features_for_each_trade_in_history"]

    # ensure we always have a test day, same logic as your current comp:contentReference[oaicite:7]{index=7}
    from_date = decrement_business_days(
        datetime.now(EASTERN).date().strftime(YEAR_MONTH_DAY), 1
    )
    prev_biz_date = get_trade_date_where_data_exists_on_this_date(from_date, data)

    (trained_model,
     test_data_date,
     prev_model,
     prev_model_date,
     encoders,
     mae,
     mae_df_list,
     intro_text) = train_model(
        data=data,
        last_trade_date=last_trade_date,
        model=model if model in ("yield_spread", "dollar_price") else "yield_spread",
        num_features_for_each_trade_in_history=n_hist_feats,
        date_for_previous_model=prev_biz_date
    )

    deploy_decision = False
    current_res_df, prev_res_df = None, None
    if trained_model is None:
        if not testing:
            send_no_new_model_email(last_trade_date, email_recipients, model)
        # Bubble failure so the pipeline marks component failed (your old flow uses this):contentReference[oaicite:8]{index=8}
        raise RuntimeError("No new data -> no new model. Exiting gracefully.")
    else:
        # same Investment-Grade MAE comparison + email you already do:contentReference[oaicite:9]{index=9}
        current_res_df, prev_res_df = mae_df_list
        try:
            curr_mae = current_res_df.loc["Investment Grade", "Mean Absolute Error"]
            prev_mae = prev_res_df.loc["Investment Grade", "Mean Absolute Error"]
            deploy_decision = curr_mae <= prev_mae
            descriptions = [
                f"New {model} model on trades of {test_data_date}.",
                f"Previous-business-day {model} model on trades of {test_data_date}."
            ]
            send_results_email_multiple_tables(
                [current_res_df, prev_res_df],
                descriptions,
                test_data_date.strftime(YEAR_MONTH_DAY),
                email_recipients,
                model,
                intro_text,
            )
        except Exception as e:
            print("Email/MAE summary issue:", e)

    # Save model as an artifact folder (same as your current comp output):contentReference[oaicite:10]{index=10}
    art = Model(uri=dsl.get_uri())
    save_path = os.path.join(art.uri, "model")
    trained_model.save(save_path)

    # pack metadata for downstream steps
    art.metadata["model_path"] = save_path
    art.metadata["deploy_decision"] = bool(deploy_decision)
    art.metadata["test_date"] = str(test_data_date)
    art.metadata["last_trade_date"] = str(last_trade_date)
    if current_res_df is not None:
        art.metadata["current_result_df"] = current_res_df.to_json(orient="index")
    if prev_res_df is not None:
        art.metadata["previous_result_df"] = prev_res_df.to_json(orient="index")
    return art


@dsl.component(
    base_image=BASE_IMAGE,
    target_image=f"gcr.io/{PROJECT_ID}/maybe_deploy_component:test"
)
def maybe_deploy_component(model_artifact: Model,
                           project_id: str,
                           region: str,
                           endpoint_display_name: str,
                           testing: bool) -> str:
    import os
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    decision = bool(model_artifact.metadata.get("deploy_decision", False))
    model_path = model_artifact.metadata.get("model_path", "")
    if not decision:
        return f"Skip deploy. deploy_decision=False. model_path={model_path}"

    # find endpoint by display name in region
    eps = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_display_name}"')
    endpoint = None
    for ep in eps:
        if ep.display_name == endpoint_display_name:
            endpoint = ep
            break
    if endpoint is None:
        raise RuntimeError(f'Endpoint with display_name="{endpoint_display_name}" not found in {region}')

    mdl = aiplatform.Model.upload(
        display_name=f"auto-{os.path.basename(model_path)}",
        artifact_uri=model_path,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-13:latest",
    )
    _ = mdl.deploy(
        endpoint=endpoint,
        traffic_percentage=100,
        machine_type="n1-standard-2"
    )
    return f'Deployed to endpoint "{endpoint_display_name}"'


# ====== pipeline ======
@dsl.pipeline(name="automated-training", pipeline_root=PIPELINE_ROOT)
def pipeline(model: str = "yield_spread_with_similar_trades",
             testing: bool = False,
             email_recipients: list = ["ficc-eng@ficc.ai"],
             learning_rate: float = 1e-4,
             batch_size: int = 1000,
             num_epochs: int = 100,
             dropout: float = 0.1,
             endpoint_display_name: str = "ficc-auto-pricing"):

    data_op = data_prep_component(model=model, testing=testing)
    set_compute_resources(data_op, cpu="64", mem="400G")

    train_op = train_eval_component(
        dataset=data_op.output,
        model=("yield_spread" if model.startswith("yield_spread") else "dollar_price"),
        testing=testing,
        email_recipients=email_recipients,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        dropout=dropout,
    )
    set_compute_resources(train_op, cpu="32", mem="200G", gpu="NVIDIA_TESLA_T4", gpu_count=2)

    deploy_op = maybe_deploy_component(
        model_artifact=train_op.output,
        project_id=PROJECT_ID,
        region=REGION,
        endpoint_display_name=endpoint_display_name,
        testing=testing,
    )
    set_compute_resources(deploy_op, cpu="2")


# ====== CLI / submit ======
def submit(enable_caching: bool, testing: bool):
    aip.init(project=PROJECT_ID, location=REGION)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    name = f"{'test-' if testing else ''}automated-training-job-{stamp}"
    job = aip.PipelineJob(
        display_name=name,
        job_id=name,
        template_path="automated-training.yaml",
        pipeline_root=PIPELINE_ROOT,
        parameter_values={},  # filled when you compile/launch from CLI options
        enable_caching=enable_caching,
        failure_policy="slow",  # keep running even if a step fails:contentReference[oaicite:14]{index=14}
    )
    job.submit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile_only", action="store_true", help="Only compile the pipeline YAML")
    parser.add_argument("--submit", action="store_true", help="Submit the pipeline to Vertex")
    parser.add_argument("--model", default="yield_spread_with_similar_trades",
                        choices=["yield_spread_with_similar_trades","dollar_price"])
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    # compile with the chosen defaults
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="automated-training.yaml",
    )

    if args.compile_only and not args.submit:
        print("Compiled to automated-training.yaml")
    else:
        # Submit with your CLI-chosen params
        aip.init(project=PROJECT_ID, location=REGION)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        name = f"{'test-' if args.testing else ''}automated-training-job-{stamp}"
        job = aip.PipelineJob(
            display_name=name,
            job_id=name,
            template_path="automated-training.yaml",
            pipeline_root=PIPELINE_ROOT,
            parameter_values={
                "model": args.model,
                "testing": args.testing,
                "email_recipients": ["gil@ficc.ai","eng@ficc.ai","eng@ficc.ai","eng@ficc.ai"],
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "dropout": args.dropout,
            },
            enable_caching=not args.no_cache,
            failure_policy="slow",
        )
        job.submit()
