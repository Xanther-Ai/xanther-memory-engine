# Contributing to Xanther Context Engine

Thank you for your interest in contributing to XCE! This document will help you get started.

## Quick Start

### Installation

```bash
# Install the package
pip install xanther-xce

# Initialize a new project
xanther init my-project
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/xanther-ai/xanther-context-engine.git
cd xanther-context-engine

# Install with dev dependencies
pip install -e ".[dev]"

# Install tree-sitter grammars (if adding new parsers)
pip install tree-sitter tree-sitter-python tree-sitter-typescript
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/parsers/test_python_parser.py

# Run with coverage
pytest --cov=xce tests/

# Run type checking
mypy xce/

# Run linting
ruff check xce/
```

## Adding New Language Parsers

XCE uses a plugin-based architecture. To add support for a new language:

1. Create a new parser file in `xce/parsers/`
2. Extend either `BaseParser` or `TreeSitterBaseParser`
3. Register it in the `ParserRegistry`

### Example: Adding a Go Parser

```python
# xce/parsers/go_parser.py
from xce.parsers.tree_sitter_base import TreeSitterBaseParser, NodeTypeMapping
from xce.parsers.registry import parser_registry

# Define node type mappings for the language
GO_NODE_MAPPING = NodeTypeMapping(
    source="source.go",
    function="function_declaration",
    class_def="type_spec",  # Go doesn't have classes in the Python sense
    method="method_declaration",
    import_statement="import_spec",
    export="export",
)

class GoParser(TreeSitterBaseParser):
    """Parser for Go source code."""
    
    language = "go"
    node_mapping = GO_NODE_MAPPING
    
    def extract_definitions(self, tree, source):
        # Implementation specific to Go AST
        ...

# Register the parser
@parser_registry.register("go")
def get_go_parser():
    return GoParser
```

### Required Methods

Your parser must implement:

- `parse(source: str) -> ParseResult` - Parse source code and return definitions/exports
- `language` - Language identifier (e.g., "python", "go")
- Optionally: `get_node_type_mapping()` - Custom node type mapping

## Code Style

We use strict type checking and linting:

```bash
# Check code style
ruff check xce/

# Fix auto-fixable issues
ruff check --fix xce/

# Run mypy with strict settings
mypy xce/ --strict

# Format code
ruff format xce/
```

### Style Guidelines

- **Type hints required** for all function signatures
- **Docstrings** for all public classes and functions
- **No bare exceptions** - catch specific exceptions
- **Use dataclasses** for structured data
- **Follow PEP 8** with 100 character line limit

## Submitting Pull Requests

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make your changes** with tests
4. **Run the full test suite**: `pytest tests/`
5. **Run type checking**: `mypy xce/ --strict`
6. **Push** to your fork
7. **Open a PR** with a clear description

### PR Requirements

- All tests must pass
- No type errors
- No linting errors
- Include tests for new functionality
- Update documentation if needed

## Project Structure

```
xanther-context-engine/
├── xce/                    # Main package
│   ├── parsers/           # Language parsers
│   ├── indexer.py         # Code indexing
│   ├── embedder.py        # Embedding generation
│   └── ...
├── tests/
│   ├── parsers/           # Parser tests
│   └── ...
├── docs/                  # Documentation
└── pyproject.toml         # Project config
```

## Getting Help

- **GitHub Issues**: https://github.com/xanther-ai/xanther-context-engine/issues
- **Discord**: Join our community
- **Email**: hello@xanther.ai

---

*By contributing, you agree to our Code of Conduct.*