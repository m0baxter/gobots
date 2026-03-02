import nltk
import torch
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_input import GenerationParameters
from lighteval.models.transformers.transformers_model import (
    TransformersModel,
    TransformersModelConfig,
)
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from transformers import AutoTokenizer, AutoModel, AutoConfig
from gobots.models.bagl import BaGLWithMTPConfig, BaGLWithMTPModel

nltk.download("punkt_tab")
torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    MODEL_NAME = "output_dir"
    # BENCHMARKS, batchsize = "gsm8k|5", 32
    # BENCHMARKS, batchsize = "truthfulqa:mc|0", 128
    # BENCHMARKS, batchsize = "piqa|0", 32 # Doesn't work
    # BENCHMARKS, batchsize = "winogrande|5", 32
    # BENCHMARKS, batchsize = "hellaswag|10", 16
    # BENCHMARKS, batchsize = "arc:challenge|0", 4
    # BENCHMARKS, batchsize = "mmlu:abstract_algebra|5,mmlu:anatomy|5,mmlu:astronomy|5,mmlu:business_ethics|5,mmlu:clinical_knowledge|5,mmlu:college_biology|5,mmlu:college_chemistry|5,mmlu:college_computer_science|5,mmlu:college_mathematics|5,mmlu:college_medicine|5,mmlu:college_physics|5,mmlu:computer_security|5,mmlu:conceptual_physics|5,mmlu:econometrics|5,mmlu:electrical_engineering|5,mmlu:elementary_mathematics|5,mmlu:formal_logic|5,mmlu:global_facts|5,mmlu:high_school_biology|5,mmlu:high_school_chemistry|5,mmlu:high_school_computer_science|5,mmlu:high_school_european_history|5,mmlu:high_school_geography|5,mmlu:high_school_government_and_politics|5,mmlu:high_school_macroeconomics|5,mmlu:high_school_mathematics|5,mmlu:high_school_microeconomics|5,mmlu:high_school_physics|5,mmlu:high_school_psychology|5,mmlu:high_school_statistics|5,mmlu:high_school_us_history|5,mmlu:high_school_world_history|5,mmlu:human_aging|5,mmlu:human_sexuality|5,mmlu:international_law|5,mmlu:jurisprudence|5,mmlu:logical_fallacies|5,mmlu:machine_learning|5,mmlu:management|5,mmlu:marketing|5,mmlu:medical_genetics|5,mmlu:miscellaneous|5,mmlu:moral_disputes|5,mmlu:moral_scenarios|5,mmlu:nutrition|5,mmlu:philosophy|5,mmlu:prehistory|5,mmlu:professional_accounting|5,mmlu:professional_law|5,mmlu:professional_medicine|5,mmlu:professional_psychology|5,mmlu:public_relations|5,mmlu:security_studies|5,mmlu:sociology|5,mmlu:us_foreign_policy|5,mmlu:virology|5,mmlu:world_religions|5", 32 # doesn't really work in latest versions

    # BENCHMARKS, batchsize = "ifeval|0", 32
    # fails:
    # BENCHMARKS = "math:algebra|4,math:counting_and_probability|4,math:geometry|4,math:intermediate_algebra|4,math:number_theory|4,math:prealgebra|4,math:precalculus|4"

    # BENCHMARKS, batchsize = "gpqa:diamond|0,gpqa:extended|0,gpqa:main|0,gpqa:mc|0", 16
    # BENCHMARKS, batchsize = "bigbench_hard:causal_judgment|3,bigbench_hard:date_understanding|3,bigbench_hard:disambiguation_qa|3,bigbench_hard:geometric_shapes|3,bigbench_hard:logical_deduction_five_objects|3,bigbench_hard:logical_deduction_seven_objects|3,bigbench_hard:logical_deduction_three_objects|3,bigbench_hard:movie_recommendation|3,bigbench_hard:navigate|3,bigbench_hard:reasoning_about_colored_objects|3,bigbench_hard:ruin_names|3,bigbench_hard:salient_translation_error_detection|3,bigbench_hard:snarks|3,bigbench_hard:sports_understanding|3,bigbench_hard:temporal_sequences|3,bigbench_hard:tracking_shuffled_objects_five_objects|3,bigbench_hard:tracking_shuffled_objects_seven_objects|3,bigbench_hard:tracking_shuffled_objects_three_objects|3", 64
    # BENCHMARKS, batchsize = "musr:murder_mysteries|0,musr:object_placements|0,musr:team_allocation|0", 16
    BENCHMARKS, batchsize = "mmlu_pro|5", 32

    evaluation_tracker = EvaluationTracker(output_dir="./results")
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        # max_samples=1,
    )

    AutoConfig.register("bagl_with_mtp", BaGLWithMTPConfig)
    AutoModel.register(BaGLWithMTPConfig, BaGLWithMTPModel)

    tokenizer = AutoTokenizer.from_pretrained("output_dir")  # , torch_dtype="bfloat16")
    model = AutoModel.from_pretrained(
        MODEL_NAME, device_map="auto"
    )  # , torch_dtype="bfloat16")
    if model.config.tie_word_embeddings:
        model._tie_or_clone_weights(
            model.bagl_model.embedding_layer, model.bagl_model.lm_head
        )
        model._tie_or_clone_weights(
            model.mtp_model.embedding_layer, model.mtp_model.lm_head
        )
        model._tie_or_clone_weights(
            model.mtp_model.embedding_layer, model.bagl_model.embedding_layer
        )
        model._tie_or_clone_weights(model.mtp_model.lm_head, model.bagl_model.lm_head)

    model = model.bagl_model
    model.eval()
    # model.compile()

    config = TransformersModelConfig(
        model_name=MODEL_NAME,
        batch_size=batchsize,
        max_length=2048,  # dtype="bfloat16"
    )  # , generation_parameters=GenerationParameters(temperature=0.2))
    model = TransformersModel.from_model(model, config)

    pipeline = Pipeline(
        model=model,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        tasks=BENCHMARKS,
    )

    results = pipeline.evaluate()
    pipeline.show_results()
    results = pipeline.get_results()
