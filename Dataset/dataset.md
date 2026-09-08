# ARCHE-Chem Dataset

Domain-specific instruction data for computational chemistry and Gaussian workflow support.

ARCHE-Chem is the specialist language model in [ARCHE](https://github.com/Arvin0313/ARCHE), an agentic system that connects mechanistic hypothesis generation with computational planning, execution and assessment. This dataset supports the specialist's tasks: understanding computational chemistry concepts, generating Gaussian route sections, interpreting calculation outputs and diagnosing execution failures.

The accompanying manuscript is **Autonomous Chemical Mechanistic Discovery through Agentic Reasoning and Validation**. Dataset construction and training are described in Supplementary Sections A2.2–A2.6; model evaluation is described in Section A5.2.

## What the data covers

| Category | Task | Content and intended use |
|---|---|---|
| **CGB** | Computational chemistry and Gaussian basics | Electronic-structure concepts, characteristics of computational methods, Gaussian input conventions, keyword meanings and calculation settings. Supports conceptual understanding and method selection. |
| **GRG** | Gaussian route generation | Route sections for tasks such as geometry optimization, frequency analysis, transition-state searches, intrinsic reaction coordinate calculations and single-point energy evaluation. Supports translating a calculation objective into Gaussian instructions. |
| **GRA** | Gaussian result analysis | Interpretation of energies, frequencies, convergence behaviour, molecular properties and reaction-path information in calculation outputs. |
| **GEH** | Gaussian error handling | Diagnosis of calculation failures and suggestions for corrections to inputs, settings or computational procedures. |

The categories combine conceptual knowledge with the application of that knowledge to specific software tasks. The dataset can be used to study domain adaptation, instruction tuning and task-level response quality in computational chemistry.

## Role in ARCHE-Chem development

ARCHE-Chem was developed from Qwen2.5-7B-Instruct through continual pretraining and supervised fine-tuning. The instruction dataset connects domain knowledge with the tasks the specialist performs within ARCHE, including Gaussian input refinement, output interpretation and error diagnosis.

The manuscript also describes a separate reward model based on Gemma-2-27B-IT for Best-of-N response selection during inference. This directory contains instruction examples and test files; the raw continual-pretraining corpus and reward-model preference data are separate resources.

## Data construction

According to the manuscript and Supplementary Information, the source material was selected to cover theoretical knowledge, software conventions and practical computational chemistry experience:

- **Textbooks and teaching materials** supplied foundational concepts and methodological explanations.
- **Gaussian documentation and input examples** supplied keyword usage, calculation specifications and software-specific conventions.
- **Expert discussions and practical research examples** supplied output interpretation, troubleshooting and workflow guidance.

The source documents were deduplicated before training. Instruction examples were manually curated and reviewed for consistency with the intended computational tasks. GRG examples draw on documented inputs and curated calculation templates; GRA and GEH examples draw on practical calculation outputs and recurring interpretation or troubleshooting questions.

## Dataset organization

Files are stored as UTF-8 JSON.

```text
dataset/
├── README.md
├── CGB_part1.json
├── CGB_part2.json
├── CGB_parh3.json
├── GRG.json
├── GRA.json
├── GEH.json
└── test_data/
    ├── test_CGB_test_part1.json
    ├── test_CGB_test_part2.json
    ├── test_CGB_test_part3.json
    ├── test_GRG.json
    ├── test_GRA.json
    └── test_GEH.json
```

CGB examples are distributed across `CGB_part1.json`, `CGB_part2.json` and `CGB_parh3.json`. The filename `CGB_parh3.json` is retained as stored in the repository. GRG, GRA and GEH examples are organized in their respective category files, with test files under `test_data/`.

| File | JSON container |
|---|---|
| `CGB_part1.json`, `CGB_parh3.json` | Object containing `computational_chemistry_qa_theory` |
| `CGB_part2.json` | Object containing `computational_chemistry_qa_gaussian` |
| `GRG.json` | Array of records |
| `GRA.json`, `GEH.json` | Object containing `gaussian_outputs` |
| Files under `test_data/` | Arrays of records |

## Record format

### CGB, GRA and GEH

Records use `question` and `answer` strings. Some CGB records also include a `Difficulty` label. That field is optional and is not provided consistently across categories or shards.

For example, a record in `CGB_part2.json` is:

```json
{
  "question": "What does the '#p' keyword in a Gaussian input file indicate?",
  "answer": "The '#p' keyword requests that Gaussian provides additional printed output, which includes more detailed information and diagnostics than the standard output.",
  "Difficulty": "Level 1"
}
```

The top-level CGB, GRA and GEH files wrap their record arrays in the keys listed above. All CGB, GRA and GEH test files are plain arrays of records, without those wrappers.

### GRG

Both `GRG.json` and `test_data/test_GRG.json` are arrays with four fields per record:

| Field | Meaning | Recommended handling |
|---|---|---|
| `question` | Natural-language calculation request | Model input |
| `ground_truth` | Stored reference Gaussian route section | Supervised target or evaluation reference |
| `basis_set` | Legacy field containing a route section, not only a basis-set name | Reference-side metadata |
| `basis_analysis` | Explanatory text associated with the stored route | Reference-side metadata |

For GRG, `question` describes the calculation objective, while `ground_truth` provides the reference route. The `basis_set` field also stores the route, and `basis_analysis` provides its accompanying explanation.

## Intended use

The dataset supports research on domain-specific instruction tuning, computational chemistry question answering, Gaussian route generation, output interpretation and error diagnosis. Software-specific answers should be interpreted in the context of the calculation method, molecular system and Gaussian configuration.

See the [project README](../README.md) for the ARCHE architecture, software setup and case-study resources.
