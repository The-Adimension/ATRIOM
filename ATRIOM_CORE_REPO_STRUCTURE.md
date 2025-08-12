```
ATRIOM/
├── README.md                          # Main repository documentation
├── LICENSE                            # Repository license
├── CONTRIBUTING.md                    # Contribution guidelines
├── CODE_OF_CONDUCT.md                # Community guidelines
├── .gitignore                         # Git ignore rules
│
├── docs/                              # Repository-wide documentation
│   ├── ATRIOM_FRAMEWORK.md           # Detailed ATRIOM methodology
│   ├── HEALTHCARE_ETHICS.md          # DEITY principles & ethics
│   └── GETTING_STARTED.md            # Quick start guide
│
├── artifacts/                         # Main collection folder
│   ├── README.md                     # Artifacts overview & index
│   │
│   ├── Gemma_3NCORE/                 # First artifact
│   │   ├── README.md                 # Artifact-specific documentation
│   │   ├── requirements.txt          # Python dependencies
│   │   ├── LICENSE                   # Artifact-specific license if different
│   │   ├── notebooks/                # Jupyter notebooks
│   │   │   └── GEMMA_3NCORE_Google_Colab_A100.ipynb
│   │   ├── src/                      # Python source files
│   │   │   └── GEMMA_3NCORE_Google_Colab_A100.py
│   │   ├── configs/                  # Configuration files
│   │   ├── docs/                     # Artifact documentation
│   │   │   ├── SETUP.md             # Setup instructions
│   │   │   ├── USAGE.md             # Usage guide
│   │   │   └── RESULTS.md           # Results & benchmarks
│   │   └── assets/                   # Images, diagrams
│   │
│   └── [Future_Artifact]/            # Template for new artifacts
│       └── ...
│
├── templates/                         # Templates for new artifacts
│   ├── ARTIFACT_README_TEMPLATE.md
│   └── artifact_structure/           # Boilerplate folder structure
│
└── utils/                            # Shared utilities
    └── common/                       # Common functions across artifacts
```