"""Service for orchestrating requirements.md and model_details.md generation."""


from neural_hub.models.database import GenerationDetails
from neural_hub.repositories.model_recommendation_repository import ModelRecommendationData
from neural_hub.utils.logger import logger


def _format_json_field(data: dict | list | None, indent: int = 0) -> str:
    """Format a JSON field as readable markdown bullet points."""
    if data is None:
        return "Not specified"

    if isinstance(data, list):
        return "\n".join(f"{'  ' * indent}- {item}" for item in data)

    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            label = key.replace("_", " ").title()
            if isinstance(value, (dict, list)):
                lines.append(f"{'  ' * indent}- **{label}**:")
                lines.append(_format_json_field(value, indent + 1))
            else:
                lines.append(f"{'  ' * indent}- **{label}**: {value}")
        return "\n".join(lines)

    return str(data)


def generate_requirements_md(details: GenerationDetails) -> str:
    """Generate requirements.md from generation_details data.

    Args:
        details: GenerationDetails ORM object with joined relationships.

    Returns:
        Markdown string for requirements.md.
    """
    logger.info(
        "Generating requirements.md",
        extra={"generation_details_id": str(details.generation_details_id)},
    )

    platform_name = details.deployment_platform.platform_name if details.deployment_platform else "Not specified"
    use_case_name = details.use_case_type.use_case_type_name if details.use_case_type else "Not specified"
    task_type_name = details.task_type.task_type_name if details.task_type else "Not specified"
    workload_name = details.workload_type.workload_type_name if details.workload_type else "Not specified"
    project_structure = details.workload_type.project_structure if details.workload_type else None

    sections = [
        f"# {details.project_name} — Requirements",
        "",
        "## Project Overview",
        f"**Project Name:** {details.project_name}",
        f"**Status:** {details.status}",
        "",
        details.project_description,
        "",
        "## Business Context",
        "",
    ]

    if details.uploaded_file_summary:
        sections.append(details.uploaded_file_summary)
    else:
        sections.append(details.project_description)

    sections += [
        "",
        "## Use Case and Task Type",
        f"- **Use Case Type:** {use_case_name}",
        f"- **Task Type:** {task_type_name}",
        f"- **Workload Type:** {workload_name}",
        "",
        "## Infrastructure Platform",
        f"- **Deployment Platform:** {platform_name}",
        "",
    ]

    # Configuration values
    sections.append("## Technical Requirements")
    if details.configuration_values:
        sections.append(_format_json_field(details.configuration_values))
    else:
        sections.append("No specific configuration values provided.")
    sections.append("")

    # Workload parameters
    sections.append("## Workload Parameters")
    if details.cost_simulation_parameter_values:
        sections.append(_format_json_field(details.cost_simulation_parameter_values))
    else:
        sections.append("No workload parameters specified.")
    sections.append("")

    # Project structure
    if project_structure:
        sections += [
            "## Project Structure",
            "",
            "```",
            project_structure,
            "```",
            "",
        ]

    content = "\n".join(sections)
    logger.info("requirements.md generated successfully")
    return content


def generate_model_details_md(data: ModelRecommendationData) -> str:
    """Generate model_details.md from aggregated model recommendation data.

    Args:
        data: Aggregated data from all model recommendation related tables.

    Returns:
        Markdown string for model_details.md.
    """
    model = data.model
    logger.info(
        "Generating model_details.md",
        extra={"model_recommendation_id": str(model.model_recommendation_id)},
    )

    sections = [
        f"# {model.model_name} — Model Details",
        "",
        "## Model Overview",
        f"- **Model Name:** {model.model_name}",
        f"- **Rank:** {model.rank}",
        f"- **Cost Category:** {model.cost_category or 'N/A'}",
        f"- **Performance:** {model.performance or 'N/A'}",
        f"- **Latency:** {model.latency or 'N/A'} ms",
        f"- **Pipeline Duration:** {model.ml_pipeline_duration or 'N/A'}",
        f"- **Pipeline Complexity:** {model.ml_pipeline_complexity or 'N/A'}",
        "",
    ]

    if model.use_case:
        sections += ["### Use Case", model.use_case, ""]
    if model.why_these_model:
        sections += ["### Why This Model", model.why_these_model, ""]
    if model.features:
        sections += ["### Features", _format_json_field(model.features), ""]

    # ML Pipeline Steps
    sections.append("## ML Pipeline")
    if data.pipeline_steps:
        for step in data.pipeline_steps:
            sections.append(f"### Step {step.step_order}: {step.step_name}")
            if step.step_description:
                sections.append(f"**Description:** {step.step_description}")
            if step.tools_used:
                sections.append(f"**Tools Used:** {_format_json_field(step.tools_used)}")
            if step.compute:
                sections.append(f"**Compute:** {_format_json_field(step.compute)}")
            if step.prerequisite_warning:
                sections.append(f"**⚠️ Prerequisite Warning:** {step.prerequisite_warning}")
            sections.append("")
    else:
        sections += ["No pipeline steps defined.", ""]

    # Hyperparameters
    sections.append("## Hyperparameters and Fine-Tuning")
    if data.hyperparameters:
        sections.append("")
        sections.append("| Parameter | Value |")
        sections.append("|-----------|-------|")
        for hp in data.hyperparameters:
            sections.append(f"| Strategy | {hp.strategy_name} |")
            if hp.learning_rate is not None:
                sections.append(f"| Learning Rate | {hp.learning_rate} |")
            if hp.epochs is not None:
                sections.append(f"| Epochs | {hp.epochs} |")
            if hp.batch_size is not None:
                sections.append(f"| Batch Size | {hp.batch_size} |")
            if hp.optimizer:
                sections.append(f"| Optimizer | {hp.optimizer} |")
            if hp.dropout is not None:
                sections.append(f"| Dropout | {hp.dropout} |")
            if hp.weight_decay is not None:
                sections.append(f"| Weight Decay | {hp.weight_decay} |")
            if hp.description:
                sections += ["", f"**Description:** {hp.description}"]
        sections.append("")
    else:
        sections += ["No hyperparameters defined.", ""]

    # Cloud Architecture
    sections.append("## Cloud Architecture")
    if data.cloud_pipeline:
        cp = data.cloud_pipeline
        for layer_name, layer_data in [
            ("Data Layer", cp.data_layer),
            ("Processing Layer", cp.processing_layer),
            ("Model Layer", cp.model_layer),
            ("API Layer", cp.api_layer),
        ]:
            sections.append(f"### {layer_name}")
            sections.append(_format_json_field(layer_data))
            sections.append("")
        if cp.architecture_notes:
            sections += ["### Architecture Notes", _format_json_field(cp.architecture_notes), ""]
    else:
        sections += ["No cloud pipeline defined.", ""]

    # Compute Recommendations
    sections.append("## Compute Recommendations")
    if data.instance_recommendations:
        sections.append("")
        sections.append("| Rank | Instance | vCPU | Memory (GB) | GPU | On-Demand ($/hr) | Spot ($/hr) |")
        sections.append("|------|----------|------|-------------|-----|------------------|-------------|")
        for inst in data.instance_recommendations:
            sections.append(
                f"| {inst.rank} | {inst.instance_name} | {inst.vcpu or 'N/A'} | "
                f"{inst.memory or 'N/A'} | {inst.gpu or 'N/A'} | "
                f"{inst.on_demand_price or 'N/A'} | {inst.spot_price or 'N/A'} |"
            )
        sections.append("")
    else:
        sections += ["No instance recommendations available.", ""]

    # Cost Analysis
    sections.append("## Cost Analysis")
    if data.cloud_cost_details:
        sections.append("")
        sections.append("| Service | Cost | Unit |")
        sections.append("|---------|------|------|")
        total_cost = 0.0
        for cd in data.cloud_cost_details:
            sections.append(f"| {cd.service_name} | ${cd.cost:.2f} | {cd.cost_unit or 'N/A'} |")
            total_cost += float(cd.cost)
        sections.append(f"| **Total** | **${total_cost:.2f}** | |")
        sections.append("")
    else:
        sections += ["No cost details available.", ""]

    # Model Analysis
    sections.append("## Model Analysis")
    if data.analysis:
        a = data.analysis
        if a.model_description:
            sections += [f"**Description:** {a.model_description}", ""]
        if a.benefits:
            sections += ["### Benefits", _format_json_field(a.benefits), ""]
        if a.risks:
            sections += ["### Risks", _format_json_field(a.risks), ""]
        sections.append("### Metrics")
        sections.append(f"- **GPU Count:** {a.gpu_count or 'N/A'}")
        sections.append(f"- **Training Time:** {a.training_time_in_hours or 'N/A'} hours")
        sections.append(f"- **Cost Per Hour:** ${a.cost_per_hour or 'N/A'}")
        sections.append(f"- **Inference Cost Per Hour:** ${a.inference_cost_per_hour or 'N/A'}")
        sections.append(f"- **Efficiency:** {a.efficiency or 'N/A'}")
        sections.append(f"- **Scalability:** {a.scalability or 'N/A'}")
        sections.append(f"- **ROI:** {a.roi or 'N/A'}")
        if a.cost_optimization_tips:
            sections += ["", "### Cost Optimization Tips", a.cost_optimization_tips]
        sections.append("")
    else:
        sections += ["No analysis data available.", ""]

    content = "\n".join(sections)
    logger.info("model_details.md generated successfully")
    return content
