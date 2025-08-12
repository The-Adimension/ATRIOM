# Contributing to ATRIOM Collection

Thank you for your interest in contributing to the ATRIOM Collection! This document provides guidelines for adding new artifacts or improving existing ones.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct, which promotes a welcoming and inclusive environment for all contributors.

## How to Contribute

### 1. Types of Contributions

We welcome:
- **New Artifacts**: ML implementations following ATRIOM methodology
- **Improvements**: Enhancements to existing artifacts
- **Documentation**: Corrections, clarifications, or translations
- **Bug Fixes**: Identifying and fixing issues
- **Features**: New capabilities for existing artifacts

### 2. Before You Start

- Check existing issues and pull requests to avoid duplication
- For major changes, open an issue first to discuss your proposal
- Ensure your contribution aligns with ATRIOM methodology and DEITY principles

### 3. Artifact Structure

New artifacts should follow this structure:
```
artifacts/YOUR_ARTIFACT_NAME/
├── README.md                 # Comprehensive documentation
├── requirements.txt          # Python dependencies
├── LICENSE                   # If different from main license
├── notebooks/               # Jupyter notebooks
├── src/                     # Python source files
├── configs/                 # Configuration files
├── docs/                    # Additional documentation
│   ├── SETUP.md
│   ├── USAGE.md
│   └── RESULTS.md
└── assets/                  # Images, diagrams
```

### 4. Artifact Requirements

Each artifact must:
- Follow the three ATRIOM phases (Reservoir, Conduit, Active)
- Include comprehensive documentation
- Provide clear setup instructions
- Be reproducible
- Include example usage
- Follow ethical AI principles

### 5. Submission Process

1. **Fork the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ATRIOM_Collections.git
   cd ATRIOM_Collections
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-artifact-name
   ```

3. **Add Your Artifact**
   - Follow the structure above
   - Ensure all dependencies are listed
   - Test thoroughly

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add [Artifact Name]: Brief description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-artifact-name
   ```
   Then create a Pull Request on GitHub

### 6. Pull Request Guidelines

Your PR should:
- Have a clear, descriptive title
- Include a detailed description of changes
- Reference any related issues
- Pass all checks (if applicable)
- Include documentation updates

### 7. Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Include docstrings for functions and classes
- Comment complex logic
- Keep code DRY (Don't Repeat Yourself)

### 8. Documentation Standards

#### README.md Template
```markdown
# Artifact Name

Brief description (1-2 sentences)

## Overview
Detailed explanation

## Quick Start
Simple usage example

## Requirements
- Hardware requirements
- Software dependencies
- Access requirements

## ATRIOM Implementation
- Reservoir Phase: ...
- Conduit Phase: ...
- Active Phase: ...

## Results
Performance metrics

## Citation
How to cite this work

## License
Licensing information
```

### 9. Testing

Before submitting:
- Test on recommended hardware
- Verify on minimal hardware
- Check all dependencies install correctly
- Ensure notebooks run without errors
- Validate outputs are as expected

### 10. Review Process

1. Maintainers will review your PR
2. You may be asked to make changes
3. Once approved, your contribution will be merged
4. You'll be added to the contributors list

## Questions?

If you have questions:
- Open an issue for public discussion
- Contact: shehab.anwer@gmail.com for private matters

## Recognition

All contributors will be:
- Listed in the repository's contributors section
- Acknowledged in relevant artifact documentation
- Credited in any publications using their contributions

Thank you for helping expand the ATRIOM Collection!
